import * as Tool from "./tool"
import DESCRIPTION from "./task.txt"
import { ToolJsonSchema } from "./json-schema"
import { BackgroundJob } from "@/background/job"
import { Session } from "@/session/session"
import { Git } from "@/git"
import { SessionID, MessageID } from "../session/schema"
import { MessageV2 } from "../session/message-v2"
import { Agent } from "../agent/agent"
import { deriveSubagentSessionPermission } from "../agent/subagent-permissions"
import type { SessionPrompt } from "../session/prompt"
import { Config } from "@/config/config"
import { Glob } from "@opencode-ai/core/util/glob"
import { Cause, Effect, Exit, Schema, Scope } from "effect"
import { EffectBridge } from "@/effect/bridge"
import { RuntimeFlags } from "@/effect/runtime-flags"
import path from "path"

export interface TaskPromptOps {
  cancel(sessionID: SessionID): Effect.Effect<void>
  resolvePromptParts(template: string): Effect.Effect<SessionPrompt.PromptInput["parts"]>
  prompt(input: SessionPrompt.PromptInput): Effect.Effect<MessageV2.WithParts>
}

const id = "task"
const fanoutID = "task_fanout"
const BACKGROUND_DESCRIPTION = [
  "",
  "",
  [
    "Background mode: background=true launches the subagent asynchronously and returns immediately.",
    "Foreground is the default; use it when you need the result before continuing.",
    "Use background only for independent work that can run while you continue elsewhere.",
    "You will be notified automatically when it finishes.",
  ].join(" "),
].join("\n")

const BaseParameterFields = {
  description: Schema.String.annotate({ description: "A short (3-5 words) description of the task" }),
  prompt: Schema.String.annotate({ description: "The task for the agent to perform" }),
  subagent_type: Schema.String.annotate({ description: "The type of specialized agent to use for this task" }),
  task_id: Schema.optional(Schema.String).annotate({
    description:
      "This should only be set if you mean to resume a previous task (you can pass a prior task_id and the task will continue the same subagent session as before instead of creating a fresh one)",
  }),
  command: Schema.optional(Schema.String).annotate({ description: "The command that triggered this task" }),
}

const BaseParameters = Schema.Struct(BaseParameterFields)

export const Parameters = Schema.Struct({
  ...BaseParameterFields,
  background: Schema.optional(Schema.Boolean).annotate({
    description: "Run the agent in the background. You will be notified when it completes.",
  }),
})

export const FanoutParameters = Schema.Struct({
  description: Schema.String.annotate({ description: "A short (3-5 words) description of the fanout task" }),
  prompt_template: Schema.String.annotate({
    description:
      "Prompt sent to each child agent. Supports {item}, {absolute_item}, {base_dir}, {index}, and {total} placeholders.",
  }),
  subagent_type: Schema.String.annotate({ description: "The child agent type to run for every item" }),
  items: Schema.optional(Schema.Array(Schema.String)).annotate({
    description: "Explicit items to process. Use include/exclude instead when the items should be discovered from files.",
  }),
  include: Schema.optional(Schema.Array(Schema.String)).annotate({
    description: "Glob patterns, relative to base_dir, used to discover file items when items is omitted.",
  }),
  exclude: Schema.optional(Schema.Array(Schema.String)).annotate({
    description: "Glob patterns, relative to base_dir, excluded after include discovery.",
  }),
  base_dir: Schema.optional(Schema.String).annotate({
    description:
      "Directory used for file discovery and absolute_item expansion. Relative paths resolve from the current session directory.",
  }),
  git_tracked: Schema.optional(Schema.Boolean).annotate({
    description: "Use git ls-files in base_dir before glob filtering. Defaults to true.",
  }),
  limit: Schema.optional(Schema.Number).annotate({ description: "Optional maximum number of items to process." }),
  max_concurrency: Schema.optional(Schema.Number).annotate({
    description:
      "Optional maximum number of child prompts to run at once. Omit it to start all child prompts concurrently.",
  }),
  reduce_prompt: Schema.optional(Schema.String).annotate({
    description:
      "Optional prompt for a final reducer child agent after all children finish. The reducer receives the fanout results.",
  }),
  reduce_subagent_type: Schema.optional(Schema.String).annotate({
    description: "Reducer agent type. Defaults to the same subagent_type.",
  }),
  command: Schema.optional(Schema.String).annotate({ description: "The command that triggered this fanout task" }),
})

type FanoutParams = Schema.Schema.Type<typeof FanoutParameters>

type FanoutEntry = {
  item: string
  absoluteItem: string
  index: number
  description: string
  prompt: string
  session?: Session.Info
  status: "pending" | "running" | "completed" | "error"
  output?: string
  error?: string
}

function output(sessionID: SessionID, text: string) {
  return [`<task id="${sessionID}" state="completed">`, "<task_result>", text, "</task_result>", "</task>"].join("\n")
}

function backgroundOutput(sessionID: SessionID) {
  return [
    `<task id="${sessionID}" state="running">`,
    "<summary>Background task started</summary>",
    "<task_result>",
    "Background task started. You will be notified automatically when it finishes; do not poll for progress.",
    "Do not duplicate its work. Continue only with non-overlapping work, or stop if there is nothing else useful to do.",
    "</task_result>",
    "</task>",
  ].join("\n")
}

function fanoutOutput(input: { total: number; failed: number; entries: FanoutEntry[]; reduce?: FanoutEntry }) {
  const children = input.entries.map((entry) =>
    [
      `<task id="${entry.session?.id ?? "unknown"}" state="${entry.status}">`,
      `<item>${entry.item}</item>`,
      entry.status === "completed" ? "<task_result>" : "<task_error>",
      entry.status === "completed" ? (entry.output ?? "") : (entry.error ?? "Unknown error"),
      entry.status === "completed" ? "</task_result>" : "</task_error>",
      "</task>",
    ].join("\n"),
  )

  return [
    `<task_fanout state="completed" total="${input.total}" failed="${input.failed}">`,
    ...children,
    input.reduce
      ? [
          `<reduce_task id="${input.reduce.session?.id ?? "unknown"}" state="${input.reduce.status}">`,
          input.reduce.status === "completed" ? "<task_result>" : "<task_error>",
          input.reduce.status === "completed" ? (input.reduce.output ?? "") : (input.reduce.error ?? "Unknown error"),
          input.reduce.status === "completed" ? "</task_result>" : "</task_error>",
          "</reduce_task>",
        ].join("\n")
      : undefined,
    "</task_fanout>",
  ]
    .filter(Boolean)
    .join("\n")
}

function backgroundMessage(input: {
  sessionID: SessionID
  description: string
  state: "completed" | "error"
  text: string
}) {
  const tag = input.state === "completed" ? "task_result" : "task_error"
  const title =
    input.state === "completed"
      ? `Background task completed: ${input.description}`
      : `Background task failed: ${input.description}`
  return [
    `<task id="${input.sessionID}" state="${input.state}">`,
    `<summary>${title}</summary>`,
    `<${tag}>`,
    input.text,
    `</${tag}>`,
    "</task>",
  ].join("\n")
}

function errorText(error: unknown) {
  if (error instanceof Error) return error.message
  return String(error)
}

function concurrency(value: number | undefined) {
  if (value === undefined) return "unbounded" as const
  const parsed = Math.floor(value)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : ("unbounded" as const)
}

function stringList(value: readonly string[] | undefined) {
  return [...(value ?? [])].filter((item) => item.trim() !== "")
}

function normalizeGlobPath(value: string) {
  return value.split(path.sep).join("/")
}

function matchesAny(base: string, item: string, patterns: string[]) {
  if (patterns.length === 0) return true
  const absolute = path.isAbsolute(item) ? item : path.resolve(base, item)
  const relative = path.relative(base, absolute) || path.basename(absolute)
  const candidates = [item, normalizeGlobPath(item), normalizeGlobPath(relative), normalizeGlobPath(absolute)]
  return patterns.some((pattern) => candidates.some((candidate) => Glob.match(pattern, candidate)))
}

function fillFanoutTemplate(
  template: string,
  entry: Pick<FanoutEntry, "item" | "absoluteItem" | "index">,
  total: number,
  baseDir: string,
) {
  const replacements = {
    item: entry.item,
    absolute_item: entry.absoluteItem,
    base_dir: baseDir,
    index: String(entry.index + 1),
    total: String(total),
  }
  let output = template
  for (const [key, value] of Object.entries(replacements)) {
    output = output.replaceAll(`{{${key}}}`, value).replaceAll(`{${key}}`, value)
  }
  return output
}

function entryDescription(description: string, item: string, index: number, total: number) {
  return `${description} ${index + 1}/${total}: ${item}`.slice(0, 180)
}

function latestAssistantText(result: MessageV2.WithParts) {
  return result.parts.findLast((item) => item.type === "text")?.text ?? ""
}

const FANOUT_DESCRIPTION = [
  "Launch one native opencode child agent per item from a single tool call.",
  "Use this when the user asks to fan out across many files/items, spawn hundreds or thousands of agents, or run a large parallel audit.",
  "Each child is a real child session with native subagent UI, inherited task permissions, cancellation, and persisted history.",
  "Use items for explicit work units, or include/exclude with base_dir to discover files. Omit max_concurrency to start all child prompts concurrently.",
].join(" ")

export const FanoutTaskTool = Tool.define(
  fanoutID,
  Effect.gen(function* () {
    const agent = yield* Agent.Service
    const config = yield* Config.Service
    const git = yield* Git.Service
    const sessions = yield* Session.Service

    const gitTrackedItems = Effect.fn("FanoutTaskTool.gitTrackedItems")(function* (baseDir: string) {
      const result = yield* git
        .run(["ls-files"], { cwd: baseDir })
        .pipe(Effect.catchCause(() => Effect.succeed(undefined)))
      if (!result || result.exitCode !== 0) return undefined
      return result
        .text()
        .split("\n")
        .map((item) => item.trim())
        .filter(Boolean)
    })

    const scanGlobItems = Effect.fn("FanoutTaskTool.scanGlobItems")(function* (
      baseDir: string,
      include: string[],
      exclude: string[],
    ) {
      const found = new Set<string>()
      for (const pattern of include) {
        const matches = yield* Effect.promise(() =>
          Glob.scan(pattern, { cwd: baseDir, dot: true, include: "file", symlink: true }),
        )
        for (const match of matches) {
          found.add(path.isAbsolute(match) ? path.relative(baseDir, match) : path.normalize(match))
        }
      }
      return [...found]
        .filter((item) => !matchesAny(baseDir, item, exclude))
        .sort((a, b) => a.localeCompare(b))
    })

    const discoverItems = Effect.fn("FanoutTaskTool.discoverItems")(function* (params: FanoutParams, parent: Session.Info) {
      const baseDir = path.resolve(parent.directory, params.base_dir ?? ".")
      const explicit = stringList(params.items)
      const include = stringList(params.include)
      const exclude = stringList(params.exclude)
      const gitTracked = params.git_tracked !== false
      const limit = params.limit === undefined ? undefined : Math.max(0, Math.floor(params.limit))

      let items = explicit
      if (items.length === 0) {
        if (include.length === 0) {
          return yield* Effect.fail(new Error("task_fanout requires either items or include patterns"))
        }
        const tracked = gitTracked ? yield* gitTrackedItems(baseDir) : undefined
        items = tracked
          ? tracked
              .filter((item) => matchesAny(baseDir, item, include))
              .filter((item) => !matchesAny(baseDir, item, exclude))
              .sort((a, b) => a.localeCompare(b))
          : yield* scanGlobItems(baseDir, include, exclude)
      }

      if (limit !== undefined) items = items.slice(0, limit)
      if (items.length === 0) return yield* Effect.fail(new Error("task_fanout found no items to process"))
      return { baseDir, include, exclude, gitTracked, items }
    })

    const run = Effect.fn("FanoutTaskTool.execute")(function* (params: FanoutParams, ctx: Tool.Context) {
      if (!ctx.extra?.bypassAgentCheck) {
        yield* ctx.ask({
          permission: id,
          patterns: [params.subagent_type, params.reduce_subagent_type].filter((item): item is string => Boolean(item)),
          always: ["*"],
          metadata: {
            description: params.description,
            subagent_type: params.subagent_type,
            fanout: true,
          },
        })
      }

      const cfg = yield* config.get()
      const ops = ctx.extra?.promptOps as TaskPromptOps
      if (!ops) return yield* Effect.fail(new Error("FanoutTaskTool requires promptOps in ctx.extra"))

      const next = yield* agent.get(params.subagent_type)
      if (!next) {
        return yield* Effect.fail(new Error(`Unknown agent type: ${params.subagent_type} is not a valid agent type`))
      }
      const reduceAgentType = params.reduce_subagent_type ?? params.subagent_type
      const reducer = params.reduce_prompt ? yield* agent.get(reduceAgentType) : undefined
      if (params.reduce_prompt && !reducer) {
        return yield* Effect.fail(new Error(`Unknown reducer agent type: ${reduceAgentType} is not a valid agent type`))
      }

      const parent = yield* sessions.get(ctx.sessionID)
      const parentAgent = parent.agent
        ? yield* agent.get(parent.agent).pipe(Effect.catchCause(() => Effect.succeed(undefined)))
        : undefined
      const discovered = yield* discoverItems(params, parent)
      const total = discovered.items.length
      const promptConcurrency = concurrency(params.max_concurrency)

      const msg = yield* MessageV2.get({ sessionID: ctx.sessionID, messageID: ctx.messageID }).pipe(Effect.orDie)
      if (msg.info.role !== "assistant") return yield* Effect.fail(new Error("Not an assistant message"))

      const fallbackModel = {
        modelID: msg.info.modelID,
        providerID: msg.info.providerID,
      }
      const childModel = next.model ?? fallbackModel
      const reducerModel = reducer?.model ?? fallbackModel

      const sessionPermission = (subagent: Agent.Info) => [
        ...deriveSubagentSessionPermission({
          parentSessionPermission: parent.permission ?? [],
          parentAgent,
          subagent,
        }),
        ...(cfg.experimental?.primary_tools?.map((item) => ({
          pattern: "*",
          action: "allow" as const,
          permission: item,
        })) ?? []),
      ]
      const childTools = (subagent: Agent.Info) => ({
        ...(subagent.permission.some((rule) => rule.permission === "todowrite") ? {} : { todowrite: false }),
        ...(subagent.permission.some((rule) => rule.permission === id) ? {} : { task: false, task_fanout: false }),
        ...Object.fromEntries((cfg.experimental?.primary_tools ?? []).map((item) => [item, false])),
      })

      const entries: FanoutEntry[] = discovered.items.map((item, index) => {
        const absoluteItem = path.isAbsolute(item) ? item : path.resolve(discovered.baseDir, item)
        const description = entryDescription(params.description, item, index, total)
        const entry = {
          item,
          absoluteItem,
          index,
          description,
          prompt: "",
          status: "pending" as const,
        }
        return {
          ...entry,
          prompt: fillFanoutTemplate(params.prompt_template, entry, total, discovered.baseDir),
        }
      })

      let reduceEntry: FanoutEntry | undefined
      const allEntries = () => (reduceEntry ? [...entries, reduceEntry] : entries)
      const counts = () => {
        const counted = allEntries()
        const completed = counted.filter((entry) => entry.status === "completed").length
        const failed = counted.filter((entry) => entry.status === "error").length
        const running = counted.filter((entry) => entry.status === "running").length
        const pending = counted.filter((entry) => entry.status === "pending").length
        return { completed, failed, running, pending }
      }
      const metadata = () => {
        const c = counts()
        return {
          parentSessionId: ctx.sessionID,
          sessionIds: allEntries().flatMap((entry) => (entry.session ? [entry.session.id] : [])),
          children: allEntries().flatMap((entry) =>
            entry.session
              ? [
                  {
                    sessionId: entry.session.id,
                    item: entry.item,
                    absoluteItem: entry.absoluteItem,
                    description: entry.description,
                    subagent_type: params.subagent_type,
                    status: entry.status,
                  },
                ]
              : [],
          ),
          total: total + (reduceEntry ? 1 : 0),
          fanoutTotal: total,
          toolCalls: total + (reduceEntry ? 1 : 0),
          completed: c.completed,
          failed: c.failed,
          running: c.running,
          pending: c.pending,
          baseDir: discovered.baseDir,
          mode: promptConcurrency === "unbounded" ? "unbounded" : "bounded",
          ...(typeof promptConcurrency === "number" ? { maxConcurrency: promptConcurrency } : {}),
        }
      }
      const updateMetadata = Effect.fn("FanoutTaskTool.metadata")(function* () {
        const c = counts()
        yield* ctx.metadata({
          title: `${params.description} ${c.completed + c.failed}/${total}`,
          metadata: metadata(),
        })
      })

      yield* Effect.forEach(
        entries,
        (entry) =>
          sessions
            .create({
              parentID: ctx.sessionID,
              title: entry.description + ` (@${next.name} subagent)`,
              permission: sessionPermission(next),
            })
            .pipe(
              Effect.tap((session) =>
                Effect.sync(() => {
                  entry.session = session
                }),
              ),
            ),
        { concurrency: "unbounded" },
      )
      yield* updateMetadata()

      const runCancel = yield* EffectBridge.make()
      function onAbort() {
        for (const entry of entries) {
          if (entry.session) runCancel.fork(ops.cancel(entry.session.id))
        }
      }

      const runEntry = Effect.fn("FanoutTaskTool.runEntry")(function* (entry: FanoutEntry) {
        if (!entry.session) return yield* Effect.fail(new Error(`Missing child session for ${entry.item}`))
        entry.status = "running"
        yield* updateMetadata()
        const parts = yield* ops.resolvePromptParts(entry.prompt)
        const result = yield* ops.prompt({
          messageID: MessageID.ascending(),
          sessionID: entry.session.id,
          model: {
            modelID: childModel.modelID,
            providerID: childModel.providerID,
          },
          agent: next.name,
          tools: childTools(next),
          parts,
        })
        entry.output = latestAssistantText(result)
        entry.status = "completed"
        yield* updateMetadata()
      })

      const runReduce = Effect.fn("FanoutTaskTool.runReduce")(function* () {
        if (!params.reduce_prompt || !reducer) return undefined
        reduceEntry = {
          item: "reduce",
          absoluteItem: discovered.baseDir,
          index: total,
          description: `${params.description} reduce`,
          prompt: [params.reduce_prompt, "", fanoutOutput({ total, failed: counts().failed, entries })].join("\n"),
          status: "pending",
        }
        reduceEntry.session = yield* sessions.create({
          parentID: ctx.sessionID,
          title: reduceEntry.description + ` (@${reducer.name} subagent)`,
          permission: sessionPermission(reducer),
        })
        reduceEntry.status = "running"
        yield* updateMetadata()
        const parts = yield* ops.resolvePromptParts(reduceEntry.prompt)
        const result = yield* ops.prompt({
          messageID: MessageID.ascending(),
          sessionID: reduceEntry.session.id,
          model: {
            modelID: reducerModel.modelID,
            providerID: reducerModel.providerID,
          },
          agent: reducer.name,
          tools: childTools(reducer),
          parts,
        })
        reduceEntry.output = latestAssistantText(result)
        reduceEntry.status = "completed"
        yield* updateMetadata()
        return reduceEntry
      })

      return yield* Effect.acquireUseRelease(
        Effect.sync(() => {
          ctx.abort.addEventListener("abort", onAbort)
        }),
        () =>
          Effect.gen(function* () {
            yield* Effect.forEach(
              entries,
              (entry) =>
                runEntry(entry).pipe(
                  Effect.catchCause((cause) =>
                    Cause.hasInterruptsOnly(cause)
                      ? Effect.failCause(cause)
                      : Effect.gen(function* () {
                          entry.status = "error"
                          entry.error = errorText(Cause.squash(cause))
                          yield* updateMetadata()
                        }),
                  ),
                ),
              { concurrency: promptConcurrency },
            )

            const reduce = yield* runReduce().pipe(
              Effect.catchCause((cause) =>
                Cause.hasInterruptsOnly(cause)
                  ? Effect.failCause(cause)
                  : Effect.succeed<FanoutEntry>({
                      item: "reduce",
                      absoluteItem: discovered.baseDir,
                      index: total,
                      description: `${params.description} reduce`,
                      prompt: params.reduce_prompt ?? "",
                      status: "error",
                      error: errorText(Cause.squash(cause)),
                    }),
              ),
            )
            const failed = counts().failed
            return {
              title: `${params.description} ${total - failed}/${total}`,
              metadata: metadata(),
              output: fanoutOutput({ total, failed, entries, reduce }),
            }
          }),
        (_, exit) =>
          Effect.gen(function* () {
            if (Exit.hasInterrupts(exit)) {
              yield* Effect.forEach(
                entries,
                (entry) => (entry.session ? ops.cancel(entry.session.id) : Effect.void),
                { concurrency: "unbounded" },
              )
            }
          }).pipe(
            Effect.ensuring(
              Effect.sync(() => {
                ctx.abort.removeEventListener("abort", onAbort)
              }),
            ),
          ),
      )
    })

    return {
      description: FANOUT_DESCRIPTION,
      parameters: FanoutParameters,
      execute: (params: FanoutParams, ctx: Tool.Context) => run(params, ctx).pipe(Effect.orDie),
    }
  }),
)

export const TaskTool = Tool.define(
  id,
  Effect.gen(function* () {
    const agent = yield* Agent.Service
    const background = yield* BackgroundJob.Service
    const config = yield* Config.Service
    const sessions = yield* Session.Service
    const scope = yield* Scope.Scope
    const flags = yield* RuntimeFlags.Service

    const run = Effect.fn("TaskTool.execute")(function* (
      params: Schema.Schema.Type<typeof Parameters>,
      ctx: Tool.Context,
    ) {
      const cfg = yield* config.get()
      const runInBackground = params.background === true
      if (runInBackground && !flags.experimentalBackgroundSubagents) {
        return yield* Effect.fail(
          new Error("Background subagents require OPENCODE_EXPERIMENTAL_BACKGROUND_SUBAGENTS=true"),
        )
      }

      if (!ctx.extra?.bypassAgentCheck) {
        yield* ctx.ask({
          permission: id,
          patterns: [params.subagent_type],
          always: ["*"],
          metadata: {
            description: params.description,
            subagent_type: params.subagent_type,
          },
        })
      }

      const next = yield* agent.get(params.subagent_type)
      if (!next) {
        return yield* Effect.fail(new Error(`Unknown agent type: ${params.subagent_type} is not a valid agent type`))
      }

      const session = params.task_id
        ? yield* sessions.get(SessionID.make(params.task_id)).pipe(Effect.catchCause(() => Effect.succeed(undefined)))
        : undefined
      const parent = yield* sessions.get(ctx.sessionID)
      const parentAgent = parent.agent
        ? yield* agent.get(parent.agent).pipe(Effect.catchCause(() => Effect.succeed(undefined)))
        : undefined
      const nextSession =
        session ??
        (yield* sessions.create({
          parentID: ctx.sessionID,
          title: params.description + ` (@${next.name} subagent)`,
          permission: [
            ...deriveSubagentSessionPermission({
              parentSessionPermission: parent.permission ?? [],
              parentAgent,
              subagent: next,
            }),
            ...(cfg.experimental?.primary_tools?.map((item) => ({
              pattern: "*",
              action: "allow" as const,
              permission: item,
            })) ?? []),
          ],
        }))

      const msg = yield* MessageV2.get({ sessionID: ctx.sessionID, messageID: ctx.messageID }).pipe(Effect.orDie)
      if (msg.info.role !== "assistant") return yield* Effect.fail(new Error("Not an assistant message"))

      const model = next.model ?? {
        modelID: msg.info.modelID,
        providerID: msg.info.providerID,
      }
      const metadata = {
        parentSessionId: ctx.sessionID,
        sessionId: nextSession.id,
        model,
        ...(runInBackground ? { background: true } : {}),
      }

      yield* ctx.metadata({
        title: params.description,
        metadata,
      })

      const ops = ctx.extra?.promptOps as TaskPromptOps
      if (!ops) return yield* Effect.fail(new Error("TaskTool requires promptOps in ctx.extra"))

      const runTask = Effect.fn("TaskTool.runTask")(function* () {
        const parts = yield* ops.resolvePromptParts(params.prompt)
        const result = yield* ops.prompt({
          messageID: MessageID.ascending(),
          sessionID: nextSession.id,
          model: {
            modelID: model.modelID,
            providerID: model.providerID,
          },
          agent: next.name,
          tools: {
            ...(next.permission.some((rule) => rule.permission === "todowrite") ? {} : { todowrite: false }),
            ...(next.permission.some((rule) => rule.permission === id) ? {} : { task: false, task_fanout: false }),
            ...Object.fromEntries((cfg.experimental?.primary_tools ?? []).map((item) => [item, false])),
          },
          parts,
        })
        return result.parts.findLast((item) => item.type === "text")?.text ?? ""
      })

      const inject = Effect.fn("TaskTool.injectBackgroundResult")(function* (
        state: "completed" | "error",
        text: string,
      ) {
        const currentParent = yield* sessions.get(ctx.sessionID)
        yield* ops
          .prompt({
            sessionID: ctx.sessionID,
            agent: currentParent.agent ?? ctx.agent,
            parts: [
              {
                type: "text",
                synthetic: true,
                text: backgroundMessage({
                  sessionID: nextSession.id,
                  description: params.description,
                  state,
                  text,
                }),
              },
            ],
          })
          .pipe(Effect.ignore, Effect.forkIn(scope, { startImmediately: true }))
      })

      const existing = yield* background.get(nextSession.id)
      if (existing?.status === "running") {
        return yield* Effect.fail(new Error(`Task ${nextSession.id} is already running.`))
      }

      if (runInBackground) {
        const info = yield* background.start({
          id: nextSession.id,
          type: id,
          title: params.description,
          metadata,
          run: runTask().pipe(
            Effect.tap((text) => inject("completed", text).pipe(Effect.ignore)),
            Effect.catchCause((cause) =>
              (Cause.hasInterruptsOnly(cause)
                ? Effect.void
                : inject("error", errorText(Cause.squash(cause))).pipe(Effect.ignore)
              ).pipe(Effect.andThen(Effect.failCause(cause))),
            ),
          ),
        })

        return {
          title: params.description,
          metadata: {
            ...metadata,
            jobId: info.id,
          },
          output: backgroundOutput(nextSession.id),
        }
      }

      const runCancel = yield* EffectBridge.make()
      const cancel = ops.cancel(nextSession.id)

      function onAbort() {
        runCancel.fork(cancel)
      }

      return yield* Effect.acquireUseRelease(
        Effect.sync(() => {
          ctx.abort.addEventListener("abort", onAbort)
        }),
        () =>
          Effect.gen(function* () {
            const text = yield* runTask()
            return {
              title: params.description,
              metadata,
              output: output(nextSession.id, text),
            }
          }),
        (_, exit) =>
          Effect.gen(function* () {
            if (Exit.hasInterrupts(exit)) yield* cancel
          }).pipe(
            Effect.ensuring(
              Effect.sync(() => {
                ctx.abort.removeEventListener("abort", onAbort)
              }),
            ),
          ),
      )
    })

    return {
      description: flags.experimentalBackgroundSubagents ? DESCRIPTION + BACKGROUND_DESCRIPTION : DESCRIPTION,
      parameters: Parameters,
      jsonSchema: flags.experimentalBackgroundSubagents ? undefined : ToolJsonSchema.fromSchema(BaseParameters),
      execute: (params: Schema.Schema.Type<typeof Parameters>, ctx: Tool.Context) =>
        run(params, ctx).pipe(Effect.orDie),
    }
  }),
)

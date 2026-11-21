# Starlings.Guardrails.Extensions.AI

Microsoft.Extensions.AI middleware that runs a [`Starlings.Guardrails`](../Starlings.Guardrails/README.md) guard around any `IChatClient`, local or hosted. **Experimental.** Depends only on `Microsoft.Extensions.AI.Abstractions`.

⭐ If this package is useful, [star Starling on GitHub](https://github.com/starlinghq/starling).

```csharp
using Microsoft.Extensions.AI;
using Starlings.Guardrails;

IChatClient client = new ChatClientBuilder(innerClient)
    .UseFunctionInvocation()                                                   // if you use tools: before the guardrails
    .Use(inner => inner.WithGuardrails(GuardLevels.Create(GuardLevel.Light)))
    .Build();

try
{
    ChatResponse response = await client.GetResponseAsync(messages);
    GuardrailsReport? report = GuardrailsChatClient.GetReport(response);   // every check, with scores and categories
}
catch (GuardrailsViolationException e)
{
    // e.Finding: the blocking check (stage, source, message index, result); e.Report: every check before it.
}
```

- **What is checked.** Before the call: the new user messages and tool results (`FunctionResultContent`), that is everything after the last assistant message, so history is not re-checked on every turn. After the call: the response text. `CheckUserMessages`, `CheckToolResults` and `CheckResponse` switch each off. Tool results are the usual route of indirect injection, so they are checked by default. A string result is checked as is; a structured one (a dictionary, a record, a `JsonElement`) as the JSON chat clients send to the model (`AIJsonUtilities.DefaultOptions`), and one that cannot be serialized is reported as not scanned.
- **Order with tool calling.** Add the guardrails *after* `UseFunctionInvocation()` (`builder.UseFunctionInvocation().Use(inner => inner.WithGuardrails(guard))`), so they sit between the tool loop and the model and see every model call, tool results included. Added before it, the whole tool loop runs below the guardrails and tool results are never checked (both orders are tested).
- **What happens.** `OnFlagged = Block` (default) throws `GuardrailsViolationException`; a flagged input never reaches the inner client. `Annotate` lets the call through. Either way the `GuardrailsReport` is in `ChatResponse.AdditionalProperties["starlings.guardrails"]` (`GuardrailsChatClient.ReportKey`): `Findings`, `Flagged`, `HasUnscanned`, `Score`.
- **Not scanned.** A text the guard could not check (for example over a model's length limit) passes and is reported as not scanned; `BlockWhenNotScanned = true` blocks it instead. The same applies to a text checked only in part, whose result lists `Gaps`: in a `Balanced` or `Strict` cascade the patterns scanned it but the model tier did not (over its length limit), or the model declined one of its questions. `HasUnscanned` covers both.
- **Streaming.** `StreamingOutputCheck.Buffer` (default) holds the updates, checks the whole response, then releases them or blocks: nothing unchecked reaches the caller, at the cost of streaming. `AfterCompletion` streams as usual and checks at the end: a block then throws after the text was shown. Both end with one extra update that carries the report (without a `MessageId`, so `updates.ToChatResponse()` keeps it in the response's `AdditionalProperties`). With `CheckResponse = false` the updates stream through and are not kept.
- Checks run one text at a time, in message order. Cancellation passes to the guard.
- **Split across messages.** `ConversationWindow` (default 4) also checks the last that many user messages, earlier turns included, joined with spaces, as one more input (`GuardSource.Conversation`, message index `GuardrailsChatClient.ConversationIndex`, -2), when the window holds two or more. One more guard check per call: negligible for patterns, one more request for a model guard; 1 turns it off. In annotate mode a flagged earlier message keeps flagging the joined text while it is in the window.
- The report is an in-process object, not a serialization contract: middleware that serializes `AdditionalProperties` (caching, logging) may not round-trip it. Put such middleware outside the guardrails or read the report before it.

## Capability status

| Capability | Status |
| --- | --- |
| Non-streaming and streaming calls, block and annotate, tool-result checks | Experimental (unit tested with a fake client and guard; composes in `ChatClientBuilder`) |
| Images, audio and other non-text content | Unsupported (not checked) |
| Checking a response while it streams (token by token) | Unsupported |
| Dependency injection helpers | Unsupported (use `ChatClientBuilder.Use`) |

## License

MIT OR Apache-2.0.

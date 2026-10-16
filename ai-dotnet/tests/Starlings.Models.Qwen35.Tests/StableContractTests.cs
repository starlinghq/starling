using Starlings.TestSupport;

namespace Starlings.Models.Qwen35.Tests;

/// <summary>
/// The capability table of each Stable profile equals its registry manifest key for key, and the package-wide table
/// agrees with it. Weight-free; runs in CI.
/// </summary>
public class StableContractTests
{
    // Per Stable profile: manifest capability -> the package-wide keys (Qwen35Capabilities.All) that carry the same status
    // on that profile (keys without a provider or variant suffix: fp32x CPU; .cuda and fp16x: the CUDA profiles).
    private static readonly Dictionary<string, Dictionary<string, string[]>> PackageKeys = new(StringComparer.Ordinal)
    {
        [Qwen35Capabilities.StableProfile] = new(StringComparer.Ordinal)
        {
            ["batch"] = ["batch"],
            ["streaming"] = ["streaming"],
            ["cancellation"] = ["cancellation.between_steps"],
            ["persistence"] = ["persistence"],
            ["training"] = ["training"],
            ["aot"] = ["aot"],
            ["extensions_ai"] = ["extensions_ai.chat_client", "extensions_ai.streaming", "extensions_ai.cancellation", "extensions_ai.usage"],
            ["mlnet"] = ["mlnet"],
            ["greedy_generation"] = ["chat.greedy.fp32x.cpu"],
            ["sampling_seeded"] = ["sampling.seeded", "extensions_ai.sampling.seeded"],
            ["stop_tokens"] = ["stop.eos_tokens"],
            ["stop_sequences"] = ["stop.sequences"],
            ["max_new_tokens"] = ["max_new_tokens"],
            ["usage"] = ["usage"],
            ["concurrent_requests"] = ["concurrent_requests.cpu", "extensions_ai.concurrent_requests"],
            ["step0_logits"] = ["step0_logits"],
            ["chat_template_owned_text"] = ["template.owned_renderer.text"],
            ["enable_thinking"] = ["template.enable_thinking"],
            ["tokenizer_owned"] = ["tokenizer.owned"],
            ["cancellation_inside_a_step"] = ["cancellation.inside_a_step"],
            ["chunked_prefill"] = ["prefill.chunked"],
            ["conversation_state_reuse"] = ["conversation_state_reuse"],
            ["tool_calling"] = ["tool_calling", "template.tools", "template.tool_calls", "template.tool_messages", "tool_call_parsing", "extensions_ai.tools", "extensions_ai.function_invocation"],
            ["reasoning_split"] = ["reasoning_split"],
            ["structured_output"] = ["structured_output", "extensions_ai.structured_output"],
            ["vision"] = ["vision", "extensions_ai.multimodal"],
            ["long_context_1667_to_8192_tokens"] = ["long_context.1667_to_8192_tokens"],
            ["long_context_over_8192_tokens"] = ["long_context.over_8192_tokens"],
        },
        [Qwen35Capabilities.StableCudaProfile] = new(StringComparer.Ordinal)
        {
            ["batch"] = ["batch"],
            ["streaming"] = ["streaming"],
            ["cancellation"] = ["cancellation.between_steps"],
            ["persistence"] = ["persistence"],
            ["training"] = ["training"],
            ["aot"] = ["aot.cuda"],
            ["extensions_ai"] = ["extensions_ai.chat_client", "extensions_ai.streaming", "extensions_ai.cancellation", "extensions_ai.usage"],
            ["mlnet"] = ["mlnet"],
            ["greedy_generation"] = ["chat.greedy.fp32x.cuda"],
            ["sampling_seeded"] = ["sampling.seeded", "extensions_ai.sampling.seeded"],
            ["stop_tokens"] = ["stop.eos_tokens"],
            ["stop_sequences"] = ["stop.sequences"],
            ["max_new_tokens"] = ["max_new_tokens"],
            ["usage"] = ["usage"],
            ["concurrent_requests"] = ["concurrent_requests.cuda.serialized", "extensions_ai.concurrent_requests"],
            ["step0_logits"] = ["step0_logits"],
            ["chat_template_owned_text"] = ["template.owned_renderer.text"],
            ["enable_thinking"] = ["template.enable_thinking"],
            ["tokenizer_owned"] = ["tokenizer.owned"],
            ["cancellation_inside_a_step"] = ["cancellation.inside_a_step"],
            ["chunked_prefill"] = ["prefill.chunked"],
            ["conversation_state_reuse"] = ["conversation_state_reuse"],
            ["tool_calling"] = ["tool_calling", "template.tools", "template.tool_calls", "template.tool_messages", "tool_call_parsing", "extensions_ai.tools", "extensions_ai.function_invocation"],
            ["reasoning_split"] = ["reasoning_split"],
            ["structured_output"] = ["structured_output", "extensions_ai.structured_output"],
            ["vision"] = ["vision", "extensions_ai.multimodal"],
            ["long_context_1667_to_8192_tokens"] = ["long_context.1667_to_8192_tokens"],
            ["long_context_over_8192_tokens"] = ["long_context.over_8192_tokens"],
        },
        [Qwen35Capabilities.StableCudaFp16xProfile] = new(StringComparer.Ordinal)
        {
            ["batch"] = ["batch"],
            ["streaming"] = ["streaming"],
            ["cancellation"] = ["cancellation.between_steps"],
            ["persistence"] = ["persistence"],
            ["training"] = ["training"],
            ["aot"] = ["aot.cuda"],
            ["extensions_ai"] = ["extensions_ai.fp16x.cuda", "extensions_ai.chat_client", "extensions_ai.streaming", "extensions_ai.cancellation", "extensions_ai.usage"],
            ["mlnet"] = ["mlnet"],
            ["greedy_generation"] = ["chat.greedy.fp16x.cuda"],
            ["sampling_seeded"] = ["sampling.seeded", "extensions_ai.sampling.seeded"],
            ["stop_tokens"] = ["stop.eos_tokens"],
            ["stop_sequences"] = ["stop.sequences"],
            ["max_new_tokens"] = ["max_new_tokens"],
            ["usage"] = ["usage"],
            ["concurrent_requests"] = ["concurrent_requests.cuda.serialized", "extensions_ai.concurrent_requests"],
            ["step0_logits"] = ["step0_logits"],
            ["chat_template_owned_text"] = ["template.owned_renderer.text"],
            ["enable_thinking"] = ["template.enable_thinking"],
            ["tokenizer_owned"] = ["tokenizer.owned"],
            ["cancellation_inside_a_step"] = ["cancellation.inside_a_step"],
            ["chunked_prefill"] = ["prefill.chunked"],
            ["conversation_state_reuse"] = ["conversation_state_reuse"],
            ["tool_calling"] = ["tool_calling", "template.tools", "template.tool_calls", "template.tool_messages", "tool_call_parsing", "extensions_ai.tools", "extensions_ai.function_invocation"],
            ["reasoning_split"] = ["reasoning_split"],
            ["structured_output"] = ["structured_output", "extensions_ai.structured_output"],
            ["vision"] = ["vision", "extensions_ai.multimodal"],
            ["long_context_1667_to_8192_tokens"] = ["long_context.1667_to_8192_tokens"],
            ["long_context_over_8192_tokens"] = ["long_context.over_8192_tokens"],
        },
    };

    private static IReadOnlyList<RegistryManifest> Family { get; } = RegistryManifest.Family(Qwen35TestAssets.RepositoryRoot, "qwen3.5-0.8b");

    private static string Name(Qwen35CapabilityStatus status) => status.ToString().ToLowerInvariant();

    [Fact]
    public void EveryStableProfileOfTheFamilyHasACodeTable()
    {
        Assert.Equal(
            Family.Where(m => m.Maturity == "stable").Select(m => m.Id).Order(StringComparer.Ordinal),
            Qwen35Capabilities.Profiles.Keys.Order(StringComparer.Ordinal));
        Assert.Equal(
            [Qwen35Capabilities.StableCudaFp16xProfile, Qwen35Capabilities.StableProfile, Qwen35Capabilities.StableCudaProfile],
            Qwen35Capabilities.Profiles.Keys.Order(StringComparer.Ordinal));
    }

    [Fact]
    public void ProfileTableEqualsTheManifestKeyForKey()
    {
        foreach (var (profile, table) in Qwen35Capabilities.Profiles)
        {
            var manifest = Family.Single(m => m.Id == profile);
            Assert.Equal(manifest.Capabilities.Keys.Order(StringComparer.Ordinal), table.Keys.Order(StringComparer.Ordinal));
            foreach (var (key, status) in manifest.Capabilities)
            {
                Assert.True(status == Name(table[key]), $"{profile} {key}: manifest {status}, Qwen35Capabilities.Profiles {Name(table[key])}");
            }

            Assert.DoesNotContain(Qwen35CapabilityStatus.Untested, table.Values);
        }
    }

    [Fact]
    public void PackageTableAgreesWithTheStableProfiles()
    {
        Assert.Equal(Qwen35Capabilities.Profiles.Keys.Order(StringComparer.Ordinal), PackageKeys.Keys.Order(StringComparer.Ordinal));
        foreach (var (profile, keys) in PackageKeys)
        {
            var table = Qwen35Capabilities.Profiles[profile];
            Assert.Equal(table.Keys.Order(StringComparer.Ordinal), keys.Keys.Order(StringComparer.Ordinal));
            foreach (var (key, packageKeys) in keys)
            {
                foreach (string packageKey in packageKeys)
                {
                    Assert.True(table[key] == Qwen35Capabilities.All[packageKey], $"{profile} {key}: profile {table[key]}, Qwen35Capabilities.All[{packageKey}] {Qwen35Capabilities.All[packageKey]}");
                }
            }

            var manifest = Family.Single(m => m.Id == profile);
            Assert.Equal("stable", manifest.Maturity);
            Assert.Equal("supported", manifest.Compatibility[profile.EndsWith("-cuda", StringComparison.Ordinal) ? "linux-x64 cuda" : "linux-x64 cpu"]);
        }

        var stable = Family.Single(m => m.Id == Qwen35Capabilities.StableProfile);
        Assert.Equal("stable", stable.Maturity);
        Assert.Equal("supported", stable.Compatibility["linux-x64 cpu"]);
        // Outside the contracts: tool calling and long context (experimental); Native AOT with the CUDA natives (experimental).
        Assert.Equal(Qwen35CapabilityStatus.Experimental, Qwen35Capabilities.All["tool_calling"]);
        Assert.Equal(Qwen35CapabilityStatus.Experimental, Qwen35Capabilities.All["long_context.1667_to_8192_tokens"]);
        Assert.Equal(Qwen35CapabilityStatus.Unsupported, Qwen35Capabilities.All["long_context.over_8192_tokens"]);
        Assert.Equal(Qwen35CapabilityStatus.Supported, Qwen35Capabilities.All["chat.greedy.fp32x.cuda"]);
        Assert.Equal(Qwen35CapabilityStatus.Supported, Qwen35Capabilities.All["chat.greedy.fp16x.cuda"]);
        Assert.Equal(Qwen35CapabilityStatus.Experimental, Qwen35Capabilities.All["aot.cuda"]);
        Assert.Equal(stable.Compatibility["windows-x64 cpu"], Name(Qwen35Capabilities.All["windows"]));
        Assert.Equal(8192, Qwen35ChatModel.MaxContextTokens);
    }
}

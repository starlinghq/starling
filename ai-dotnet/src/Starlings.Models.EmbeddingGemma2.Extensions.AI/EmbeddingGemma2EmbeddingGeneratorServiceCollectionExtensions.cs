using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;

namespace Starlings.Models.EmbeddingGemma2.Extensions.AI;

/// <summary>Registers an <see cref="EmbeddingGemma2EmbeddingGenerator"/> in a service collection.</summary>
public static class EmbeddingGemma2EmbeddingGeneratorServiceCollectionExtensions
{
    /// <summary>
    /// Registers one <see cref="EmbeddingGemma2EmbeddingGenerator"/> singleton, created from <paramref name="files"/>
    /// (graph and owned tokenizer) on first resolution, and <c>IEmbeddingGenerator&lt;string, Embedding&lt;float&gt;&gt;</c>
    /// resolving to the same instance unless one is already registered.
    /// </summary>
    public static IServiceCollection AddEmbeddingGemma2EmbeddingGenerator(
        this IServiceCollection services, EmbeddingGemma2ModelFiles files, EmbeddingGemma2EmbeddingGeneratorOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(files);
        (options ?? new EmbeddingGemma2EmbeddingGeneratorOptions()).Validate();
        return services.AddEmbeddingGemma2EmbeddingGenerator(_ => EmbeddingGemma2EmbeddingGenerator.Create(files, options));
    }

    /// <summary>
    /// Registers one <see cref="EmbeddingGemma2EmbeddingGenerator"/> singleton, created from <paramref name="files"/> and <paramref name="tokenizer"/>
    /// on first resolution, and <c>IEmbeddingGenerator&lt;string, Embedding&lt;float&gt;&gt;</c> resolving to the same
    /// instance unless one is already registered. The container owns the generator, which owns the encoder and session.
    /// </summary>
    public static IServiceCollection AddEmbeddingGemma2EmbeddingGenerator(
        this IServiceCollection services, EmbeddingGemma2ModelFiles files, IEmbeddingGemma2Tokenizer tokenizer, EmbeddingGemma2EmbeddingGeneratorOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(files);
        ArgumentNullException.ThrowIfNull(tokenizer);
        (options ?? new EmbeddingGemma2EmbeddingGeneratorOptions()).Validate();
        return services.AddEmbeddingGemma2EmbeddingGenerator(_ => EmbeddingGemma2EmbeddingGenerator.Create(files, tokenizer, options));
    }

    /// <summary>
    /// Registers one <see cref="EmbeddingGemma2EmbeddingGenerator"/> singleton from <paramref name="factory"/> and
    /// <c>IEmbeddingGenerator&lt;string, Embedding&lt;float&gt;&gt;</c> resolving to the same instance unless one is
    /// already registered. The container disposes the instance the factory returns.
    /// </summary>
    public static IServiceCollection AddEmbeddingGemma2EmbeddingGenerator(this IServiceCollection services, Func<IServiceProvider, EmbeddingGemma2EmbeddingGenerator> factory)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(factory);
        services.AddSingleton(factory);
        services.TryAddSingleton<IEmbeddingGenerator<string, Embedding<float>>>(sp => sp.GetRequiredService<EmbeddingGemma2EmbeddingGenerator>());
        return services;
    }

    /// <summary>
    /// Registers one <see cref="EmbeddingGemma2MultimodalEmbeddingGenerator"/> singleton, created from
    /// <paramref name="files"/> on first resolution, and <c>IEmbeddingGenerator&lt;DataContent, Embedding&lt;float&gt;&gt;</c>
    /// and <c>IEmbeddingGenerator&lt;IEnumerable&lt;AIContent&gt;, Embedding&lt;float&gt;&gt;</c> resolving to the same
    /// instance unless already registered. Set <see cref="EmbeddingGemma2MultimodalEmbeddingGeneratorOptions.ImageDecoder"/>
    /// to accept encoded images.
    /// </summary>
    public static IServiceCollection AddEmbeddingGemma2MultimodalEmbeddingGenerator(
        this IServiceCollection services, EmbeddingGemma2ImageModelFiles files, EmbeddingGemma2MultimodalEmbeddingGeneratorOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(files);
        (options ?? new EmbeddingGemma2MultimodalEmbeddingGeneratorOptions()).Validate();
        services.AddSingleton(_ => EmbeddingGemma2MultimodalEmbeddingGenerator.Create(files, options));
        services.TryAddSingleton<IEmbeddingGenerator<DataContent, Embedding<float>>>(sp => sp.GetRequiredService<EmbeddingGemma2MultimodalEmbeddingGenerator>());
        services.TryAddSingleton<IEmbeddingGenerator<IEnumerable<AIContent>, Embedding<float>>>(sp => sp.GetRequiredService<EmbeddingGemma2MultimodalEmbeddingGenerator>());
        return services;
    }
}

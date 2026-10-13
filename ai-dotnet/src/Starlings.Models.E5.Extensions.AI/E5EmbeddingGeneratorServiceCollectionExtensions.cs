using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;

namespace Starlings.Models.E5.Extensions.AI;

/// <summary>Registers an <see cref="E5EmbeddingGenerator"/> in a service collection.</summary>
public static class E5EmbeddingGeneratorServiceCollectionExtensions
{
    /// <summary>
    /// Registers one <see cref="E5EmbeddingGenerator"/> singleton, created by <see cref="E5EmbeddingGenerator.Create"/>
    /// from <paramref name="files"/> on first resolution, and <c>IEmbeddingGenerator&lt;string, Embedding&lt;float&gt;&gt;</c>
    /// resolving to the same instance unless one is already registered.
    /// </summary>
    /// <remarks>
    /// The container owns the generator, which owns the encoder and the ONNX Runtime session: disposing the service
    /// provider releases them. To add Microsoft.Extensions.AI middleware, register the pipeline after this call, for
    /// example <c>services.AddEmbeddingGenerator(sp =&gt; sp.GetRequiredService&lt;E5EmbeddingGenerator&gt;()).UseLogging()</c>;
    /// the later registration wins, and disposing the pipeline disposes the generator, which is idempotent.
    /// </remarks>
    public static IServiceCollection AddE5EmbeddingGenerator(this IServiceCollection services, E5ModelFiles files, E5EmbeddingGeneratorOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(files);
        (options ?? new E5EmbeddingGeneratorOptions()).Validate();
        return services.AddE5EmbeddingGenerator(_ => E5EmbeddingGenerator.Create(files, options));
    }

    /// <summary>
    /// Registers one <see cref="E5EmbeddingGenerator"/> singleton from <paramref name="factory"/> and
    /// <c>IEmbeddingGenerator&lt;string, Embedding&lt;float&gt;&gt;</c> resolving to the same instance unless one is
    /// already registered. The container disposes the instance the factory returns.
    /// </summary>
    public static IServiceCollection AddE5EmbeddingGenerator(this IServiceCollection services, Func<IServiceProvider, E5EmbeddingGenerator> factory)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(factory);
        services.AddSingleton(factory);
        services.TryAddSingleton<IEmbeddingGenerator<string, Embedding<float>>>(sp => sp.GetRequiredService<E5EmbeddingGenerator>());
        return services;
    }
}

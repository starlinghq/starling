using System.Reflection;

namespace Starlings.Models.Kev.Tests;

/// <summary>
/// The model definition package (Starlings.Models.Kev): request rendering and encoding without ONNX Runtime; the decision
/// runtime is Starlings.Models.Kev.Onnx in the same namespace.
/// </summary>
public sealed class DefinitionTests
{
    [Fact]
    public void DefinitionAssemblyReferencesNoOnnxRuntime()
    {
        AssemblyName[] references = typeof(KevRequest).Assembly.GetReferencedAssemblies();
        Assert.DoesNotContain(references, r => r.Name!.StartsWith("Microsoft.ML.OnnxRuntime", StringComparison.Ordinal));
        Assert.DoesNotContain(references, r => r.Name is "Starlings.Runtime.Onnx" or "Starlings.Acquisition");
        Assert.Same(typeof(KevRequest).Assembly, typeof(KevEncoder).Assembly);
        Assert.Same(typeof(KevRequest).Assembly, typeof(KevText).Assembly);
        Assert.Same(typeof(KevRequest).Assembly, typeof(KevModelProfile).Assembly);
        Assert.Equal("Starlings.Models.Kev.Onnx", typeof(KevDecisionModel).Assembly.GetName().Name);
        Assert.Equal(typeof(KevRequest).Namespace, typeof(KevDecisionModel).Namespace);
    }

    [Fact]
    public void EncoderIsPublic() => Assert.True(typeof(KevEncoder).IsPublic);
}

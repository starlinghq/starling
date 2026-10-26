// Weight-free Native AOT smoke consumer for the Starlings packages (tools/aot-smoke-ai.sh).
// The project roots the whole package assembly, so the published binary carries it. This program loads the assembly
// by name and prints one line the script checks; it runs no inference and uses no reflection that trimming could
// break, so every warning of the publish belongs to the package or its dependencies, none to this program.
//
//   AotSmoke
//
// Exit code 0 when the package assembly loads.
using System.Reflection;
using System.Runtime.CompilerServices;

string package = typeof(Marker).Assembly.GetCustomAttributes<AssemblyMetadataAttribute>()
    .Single(attribute => attribute.Key == "SmokePackage").Value!;
var assembly = Assembly.Load(new AssemblyName(package));
Console.WriteLine($"mode:    {(RuntimeFeature.IsDynamicCodeSupported ? "JIT" : "Native AOT")}");
Console.WriteLine($"package: {assembly.GetName().Name} {assembly.GetName().Version}");
return assembly.GetName().Name == package ? 0 : 1;

/// <summary>Anchor for this assembly's metadata.</summary>
internal sealed class Marker;

# Builds the E5Search console sample into a Linux image with no Python.
#
#   docker build -t starling-e5search .
#   docker run --rm -v starling-models:/models starling-e5search download   # once; fills the volume
#   docker run --rm -v starling-models:/models starling-e5search search "Warum rostet Eisen?"
#
# The image has no Python: the base is mcr.microsoft.com/dotnet/runtime-deps:10.0-noble-chiseled
# (Ubuntu chiseled, distroless-style: no shell, no package manager, no Python). ONNX Runtime's CPU
# natives need libstdc++6, libgcc-s1 and libgomp1, which the runtime-deps chiseled image ships.
# Multi-stage: the SDK image builds and publishes, nothing but the published output and the chiseled
# runtime enters the final layer.
FROM mcr.microsoft.com/dotnet/sdk:10.0-noble AS build
WORKDIR /src
COPY E5Search.csproj ./
RUN dotnet restore E5Search.csproj
COPY Program.cs ./
RUN dotnet publish E5Search.csproj -c Release -o /app --no-restore

# runtime (not runtime-deps): the chiseled runtime image ships the .NET shared runtime plus the
# native loader deps ONNX Runtime needs (libstdc++6, libgcc-s1, libgomp1), and no Python, no shell,
# no package manager. runtime-deps alone has no dotnet executable.
FROM mcr.microsoft.com/dotnet/runtime:10.0-noble-chiseled AS final
WORKDIR /app
COPY --from=build /app .
# Models come from a volume, not the image: mount -v starling-models:/models and pass --cache /models.
ENV STARLING_MODELS=/models
ENTRYPOINT ["dotnet", "E5Search.dll"]

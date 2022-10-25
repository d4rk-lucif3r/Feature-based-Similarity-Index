import os

os.environ["TFY_HOST"] = "https://app.truefoundry.com/"
os.environ["TFY_API_KEY"] = "<your-api-key>"  # replace this

from servicefoundry import Build, PythonBuild, Resources, Service

service = Service(
    name="image-similarity",
    image=Build(
        build_spec=PythonBuild(
            command="python app.py",
        ),
    ),
    ports=[{"port": 8080}],
    resources=Resources(memory_limit=1000, memory_request=500,
                        cpu_limit=2, cpu_request=1.5),
)
service.deploy(workspace_fqn="<your-fqn>")

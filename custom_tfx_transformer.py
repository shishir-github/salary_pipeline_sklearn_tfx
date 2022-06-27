
import tfx

class HelloComponentSpec(tfx.types.ComponentSpec):
  """ComponentSpec for Custom TFX Hello World Component."""

  PARAMETERS = {

  }
  INPUTS = {
      'examples': tfx.ChannelParameter(type=standard_artifacts.Examples),
  }
  OUTPUTS = {
      'augmented_data': tfx.ChannelParameter(type=standard_artifacts.Examples),
  }


  class Executor(base_executor.BaseExecutor):
  """Executor for HelloComponent."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    ...

    split_to_instance = {}
    for artifact in input_dict['input_data']:
      for split in json.loads(artifact.split_names):
        uri = artifact_utils.get_split_uri([artifact], split)
        split_to_instance[split] = uri

    for split, instance in split_to_instance.items():
      input_dir = instance
      output_dir = artifact_utils.get_split_uri(
          output_dict['output_data'], split)
      for filename in tf.io.gfile.listdir(input_dir):
        input_uri = os.path.join(input_dir, filename)
        output_uri = os.path.join(output_dir, filename)
        io_utils.copy_file(src=input_uri, dst=output_uri, overwrite=True)




from tfx.types import standard_artifacts
from hello_component import executor

class HelloComponent(base_component.BaseComponent):
  """Custom TFX Hello World Component."""

  SPEC_CLASS = HelloComponentSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(self,
               input_data: types.Channel = None,
               output_data: types.Channel = None,
               name: Optional[Text] = None):
    if not output_data:
      examples_artifact = standard_artifacts.Examples()
      examples_artifact.split_names = input_data.get()[0].split_names
      output_data = channel_utils.as_channel([examples_artifact])

    spec = HelloComponentSpec(input_data=input_data,
                              output_data=output_data, name=name)
    super(HelloComponent, self).__init__(spec=spec)
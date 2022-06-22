import os
from typing import List, Text

import absl
import tensorflow_model_analysis as tfma
from tfx.orchestration import pipeline
from tfx.proto.orchestration import pipeline_pb2
import tfx
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2


_pipeline_name = 'salary__pipeline_sklearn_local'

_root = os.path.dirname(__file__)

_data_root = os.path.join(_root, 'data')

_transform_module_file = os.path.join(_root, 'data_transform.py')

_trainer_module_file = os.path.join(_root, 'salary_utils_sklearn.py')

_evaluator_module_file = os.path.join(_root,
                                      'sklearn_predict_extractor.py')

_serving_model_dir = os.path.join(_root, 'serving_model',
                                  _pipeline_name)

_tfx_root = os.path.join(_root, 'tfx')

_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)

# Sqlite ML-metadata db path.
_metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name,
                              'metadata.db')

_beam_pipeline_args = [
    '--direct_running_mode=multi_threading',
    # 0 means auto-detect based on on the number of CPUs available
    # during execution time.
    '--direct_num_workers=1',
]


def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    data_root: Text,
    trainer_module_file: Text,
    #evaluator_module_file: Text,
    serving_model_dir: Text,
    metadata_path: Text,
    beam_pipeline_args: List[Text],
     ) -> pipeline.Pipeline:


  example_gen = tfx.components.CsvExampleGen(input_base=data_root)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = tfx.components.StatisticsGen(
      examples=example_gen.outputs['examples'])

  # Generates schema based on statistics files.
  schema_gen = tfx.components.SchemaGen(
      statistics=statistics_gen.outputs['statistics'],
      infer_feature_shape=True)

  # Performs anomaly detection based on statistics and data schema.
  example_validator = tfx.components.ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema'])

  trainer = tfx.components.Trainer(
      module_file=trainer_module_file,
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      train_args=trainer_pb2.TrainArgs(num_steps=2000),
      #eval_args=trainer_pb2.EvalArgs()
      )

  pusher = tfx.components.Pusher(model=trainer.outputs['model'],
      #model_blessing=evaluator.outputs['blessing'],
      push_destination=pusher_pb2.PushDestination(
      filesystem=pusher_pb2.PushDestination.Filesystem(
      base_directory=serving_model_dir)))

  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[
          example_gen,
          statistics_gen,
          schema_gen,
          example_validator,
          trainer,
          #model_resolver,
          #evaluator,
          pusher,
      ],
      enable_cache=True,
      metadata_connection_config=tfx.orchestration.metadata.
      sqlite_metadata_connection_config(metadata_path),
      beam_pipeline_args=beam_pipeline_args,
  )


# To run this pipeline from the python CLI:
#   $python penguin_pipeline_sklearn_local.py
if __name__ == '__main__':
  absl.logging.set_verbosity(absl.logging.INFO)
  LocalDagRunner().run(
      create_pipeline(pipeline_name=_pipeline_name,
                       pipeline_root=_pipeline_root,
                       data_root=_data_root,
                       trainer_module_file=_trainer_module_file,
                       #evaluator_module_file=_evaluator_module_file,
                       serving_model_dir=_serving_model_dir,
                       metadata_path=_metadata_path,
                       beam_pipeline_args=_beam_pipeline_args))

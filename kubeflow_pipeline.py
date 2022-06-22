import os 
from absl import logging 
from tfx.orchestration.kubeflow import kubeflow_dag_runner 
from penguin_pipeline_sklearn_local import create_pipeline
import time 
suffix = int(time.time())

_pipeline_name = 'penguin_sklearn_local'

# This example assumes that the Penguin example is the working directory. Feel
# free to customize as needed.
_penguin_root = os.path.dirname(__file__)
_data_root = os.path.join(_penguin_root, 'data')

# Python module file to inject customized logic into the TFX components.
# Trainer requires user-defined functions to run successfully.
_trainer_module_file = os.path.join(_penguin_root, 'penguin_utils_sklearn.py')

# Python module file to inject customized logic into the TFX components. The
# Evaluator component needs a custom extractor in order to make predictions
# using the scikit-learn model.
_evaluator_module_file = os.path.join(_penguin_root,
                                      'sklearn_predict_extractor.py')

# Path which can be listened to by the model server.  Pusher will output the
# trained model here.
_serving_model_dir = os.path.join(_penguin_root, 'serving_model',
                                  _pipeline_name)

# Directory and data locations.  This example assumes all of the
# example code and metadata library is relative to $HOME, but you can store
# these files anywhere on your local filesystem.
_tfx_root = os.path.join(_penguin_root, 'tfx')
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
# Sqlite ML-metadata db path.
_metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name,
                              'metadata.db')

# Pipeline arguments for Beam powered Components.
# TODO(b/171316320): Change direct_running_mode back to multi_processing and set
# direct_num_workers to 0.
_beam_pipeline_args = [
    '--direct_running_mode=multi_threading',
    # 0 means auto-detect based on on the number of CPUs available
    # during execution time.
    '--direct_num_workers=1',
]

def run():
    
    metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()
    #tfx_image = 'gcr.io/singular-willow-339022/mlimage'
    runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        kubeflow_metadata_config=metadata_config
    )

    kubeflow_dag_runner.KubeflowDagRunner(config=runner_config).run(
        create_pipeline(
                        pipeline_name=_pipeline_name,
                       pipeline_root=_pipeline_root,
                       data_root=_data_root,
                       trainer_module_file=_trainer_module_file,
                       evaluator_module_file=_evaluator_module_file,
                       serving_model_dir=_serving_model_dir,
                       metadata_path=_metadata_path,
                       beam_pipeline_args=_beam_pipeline_args)
    )



      






if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    run()

#tfx pipeline create --pipeline-path=kubeflow_pipeline.py --endpoint=http://localhost:8081
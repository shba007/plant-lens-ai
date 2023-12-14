# import tensorflowjs as tfjs
import subprocess

ROOT_PATH = "D:/Programming/Projects/Public/plant-lens/ai"
DEVELOPMENT_MODEL_PATH = f"{ROOT_PATH}/model/development"
DEPLOYMENT_MODEL_PATH = f"{ROOT_PATH}/model/deployment"
VERSION_TAG = "0.1.0-37"  # input("Enter Version Tag (e.g 0.0.0-0):")


def main():
    # model = tf.keras.models.load_model(
    # f"{DEVELOPMENT_MODEL_PATH}/v{VERSION_TAG}")
    # model.summary()
    # tfjs.converters.save_keras_model(model, "tfjs_model")
    command = f"tensorflowjs_converter --input_format=keras --output_format=tfjs_layers_model {DEVELOPMENT_MODEL_PATH}/v{VERSION_TAG} {DEPLOYMENT_MODEL_PATH}/v{VERSION_TAG}"
    print(command)

    # Run the command
    result = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Print the result
    print("Exit code:", result.returncode)
    print("Standard output:")
    print(result.stdout)
    print("Standard error:")
    print(result.stderr)


if __name__ == "__main__":
    main()

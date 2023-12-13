import subprocess

ROOT_PATH = "D:/Programming/Projects/Public/plant-lens/ai"
DEVELOPMENT_MODEL_PATH = f"{ROOT_PATH}/model/development"
DEPLOYMENT_MODEL_PATH = f"{ROOT_PATH}/model/deployment"
VERSION_TAG = "0.0.0-51"  # input("Enter Version Tag (e.g 0.0.0-0):")


def main():
    command = f"tensorflowjs_converter --input_format=tf_saved_model --saved_model_tags=serve {DEVELOPMENT_MODEL_PATH}/v{VERSION_TAG} {DEPLOYMENT_MODEL_PATH}/v{VERSION_TAG}"
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

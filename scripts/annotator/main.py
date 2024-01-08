import os
import math

ROOT_PATH = "D:/Programming/Projects/Public/plant-lens/ai"
ANNOTATED_DATA_PATH = f"{ROOT_PATH}/data/annotated"


def generate_name(counter):
    pad = 3 - math.floor(math.log(counter, 10) + 0.000001)
    counter = f'{"0"*pad}{counter}'
    return f"img-{counter}.jpg"


def main():
    # rename every file in a format img-$$
    for dir in os.listdir(ANNOTATED_DATA_PATH):
        counter = 1
        for file in os.listdir(f"{ANNOTATED_DATA_PATH}/{dir}"):
            # print(generate_name(counter))
            os.rename(f"{ANNOTATED_DATA_PATH}/{dir}/{file}",
                      f"{ANNOTATED_DATA_PATH}/{dir}/{generate_name(counter)}")
            counter += 1


if __name__ == "__main__":
    main()

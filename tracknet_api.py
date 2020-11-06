import argparse
from predict_video_api import predict_video
from ground_truth_generator_api import ground_truth_generator
from train_api import train

#parse parameters
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type = str, default = "Unknown")
parser.add_argument("--input_video_path", type=str)
parser.add_argument("--output_video_path", type=str)
parser.add_argument("--save_weights_path", type = str)
parser.add_argument("--save_pridict_list_path", type = str)
parser.add_argument("--n_classes", type=int, default = 256)

parser.add_argument("--ground_truth_generator_img_path", type = str)

parser.add_argument("--training_images_name", type = str)
parser.add_argument("--input_height", type=int , default = 360)
parser.add_argument("--input_width", type=int , default = 640)
parser.add_argument("--epochs", type = int, default = 1000)
parser.add_argument("--batch_size", type = int, default = 1)
parser.add_argument("--load_weights", type = str , default = "-1")
parser.add_argument("--step_per_epochs", type = int, default = 200 )

args = parser.parse_args()
mode = args.mode
input_video_path = args.input_video_path
output_video_path = args.output_video_path
save_weights_path = args.save_weights_path
n_classes =  args.n_classes

img_path = args.ground_truth_generator_img_path

training_images_name = args.training_images_name
input_height = args.input_height
input_width = args.input_width
epochs = args.epochs
train_batch_size = args.batch_size
load_weights = args.load_weights
step_per_epochs = args.step_per_epochs

if mode == "predict_video":
    predict_video = predict_video()
    predict_video.predict_video(input_video_path, output_video_path, save_weights_path, n_classes)
elif mode == "ground_truth_generator":
    ground_truth_generator = ground_truth_generator()
    ground_truth_generator.ground_truth_generator(img_path)
elif mode == "train":
    train = train()
    train.train(save_weights_path, training_images_name, n_classes, input_height, input_width, epochs, train_batch_size, load_weights, step_per_epochs)
else:
    print("Unknown")
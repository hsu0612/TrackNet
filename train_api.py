import Models , LoadBatches
from keras import optimizers
from keras.utils import plot_model

class train():
    def __init__(self):
        self.optimizer_name = optimizers.Adadelta(lr=1.0)
    def train(self, save_weights_path, training_images_name, n_classes, input_height, input_width, epochs, train_batch_size, load_weights, step_per_epochs):
        #load TrackNet model
        modelTN = Models.TrackNet.TrackNet
        m = modelTN( n_classes , input_height=input_height, input_width=input_width)
        m.compile(loss='categorical_crossentropy', optimizer=self.optimizer_name, metrics=['accuracy'])

        #check if need to retrain the model weights
        if load_weights != "-1":
            m.load_weights("./weights/model." + load_weights)

        #show TrackNet details, save it as TrackNet.png
        plot_model( m , show_shapes=True , to_file='TrackNet.png')

        #get TrackNet output height and width
        model_output_height = m.outputHeight
        model_output_width = m.outputWidth

        #creat input data and output data
        Generator  = LoadBatches.InputOutputGenerator( training_images_name,  train_batch_size,  n_classes , input_height , input_width , model_output_height , model_output_width)

        #start to train the model, and save weights per 50 epochs  

        for ep in range(1, epochs+1 ):
            print("Epoch :", str(ep) + "/" + str(epochs))
            m.fit_generator(Generator, step_per_epochs, workers = 0)
            #if ep % 5 == 0:
            m.save_weights(save_weights_path + ".0")

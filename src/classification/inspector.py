import os
import random

import cv2
from keras import backend as K


class Inspector(object):

    LAYERS = ['block1_conv1', 'block1_conv2', 'block1_pool',
              'block2_conv1', 'block2_conv2', 'block2_pool',
              'block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_conv4','block3_pool',
              'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_conv4','block4_pool',
              'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_conv4','block5_pool', ]
              #'fc1', 'fc2']
              
    def __init__(self, model):
        self.model = model
        self.functors = dict()

        inp = model.input
        for name in self.LAYERS:
            self.functors[name] = K.function([inp], [model.get_layer(name).output])

    def visualize(self, images):

        j = 0
        for img in images:
            #cv2.imshow('input', img)

            os.mkdir('tmp/{}'.format(j))
            cv2.imwrite('tmp/{}/input.png'.format(j), img)
            for name in self.functors:
                #if name != 'block3_pool':
                    #continue

                res = self.functors[name]([[img]])[0]

                for i in random.sample(range(res.shape[-1]), 5):
                    layer_out = res[0, :, :,i].astype('uint8')
                    print(name, i, layer_out.shape)
                    #cv2.imshow('{}-{}'.format(name, i), cv2.resize(layer_out, (200, 200), interpolation=cv2.INTER_NEAREST))

                    cv2.imwrite('tmp/{}/{}-{}.png'.format(j, name, i), cv2.resize(layer_out, (200, 200), interpolation=cv2.INTER_NEAREST))

            j += 1
            if j == 5: exit()
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()


import keras
import numpy as np
import pprint

model = keras.models.load_model("./ssd_face_cfg2.hdf5")
config = model.get_config()
pprint.pprint(config)


bn_config = {'class_name': 'BatchNormalization',
             'config': {'axis': 3,
                        'beta_constraint': None,
                        'beta_initializer': {'class_name': 'Zeros',
                                             'config': {}},
                        'beta_regularizer': None,
                        'center': True,
                        'epsilon': 0.0,
                        'gamma_constraint': None,
                        'gamma_initializer': {'class_name': 'Ones',
                                              'config': {}},
                        'gamma_regularizer': None,
                        'momentum': 0.99,
                        'moving_mean_initializer': {'class_name': 'Zeros',
                                                    'config': {}},
                        'moving_variance_initializer': {'class_name': 'Ones',
                                                        'config': {}},
                        'name': 'bn0',
                        'scale': True,
                        'trainable': False},
             'inbound_nodes': [[['input_1', 0, 0, {}]]],
             'name': 'bn0'}

config['layers'].insert(1, bn_config)
config['layers'][2]['inbound_nodes'] = [[['bn0', 0, 0, {}]]]

new_model = keras.models.Model.from_config(config)
new_model.load_weights("./ssd_face_cfg2.hdf5", by_name=True)

bn0 = new_model.get_layer('bn0')
weights = bn0.get_weights()
bn0.set_weights([np.ones_like(weights[0])*2, *weights[1:]])

# print(weights)

inputs = np.random.random((1, 200, 200, 3))

res1 = model.predict((inputs - 127.5)/127.5)
res2 = new_model.predict(inputs / 255. - 0.5)

new_model.save("ssd_bn2.hdf5")
print([(np.sum(np.abs(ele1 - ele2)), ele1.shape) for ele1, ele2 in zip(res1, res2)])
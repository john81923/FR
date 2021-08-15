import keras
import pprint
import numpy as np

old_model = keras.models.load_model("ssd_face_cfg2_smallbox.hdf5")
config = old_model.get_config()

# pprint.pprint(config)

config['layers'].pop(1)
config['layers'][1]['inbound_nodes'] = [[['input_1', 0, 0, {}]]]

new_model = keras.models.Model.from_config(config)
new_model.load_weights("ssd_face_cfg2_smallbox.hdf5", by_name=True)

for _ in range(5):
	img = np.random.random((1, 200, 200, 3))*255
	pred_old = old_model.predict(img/255. - 0.5)
	pred_new = new_model.predict(img/127.5 - 1.)
	print([(np.sum(np.abs(ele1 - ele2)), ele1.shape) for ele1, ele2 in zip(pred_old, pred_new)])

new_model.save("ssd_face_cfg2_smallbox_noBN.hdf5")
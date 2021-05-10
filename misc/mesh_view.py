import scipy.io
import trimesh

data = scipy.io.loadmat("cachedir/cub/sfm/anno_train.mat")
print(data["S"].shape, data["conv_tri"].shape)
mesh = trimesh.Trimesh(data["S"].T, data["conv_tri"] - 1)
mesh.show()

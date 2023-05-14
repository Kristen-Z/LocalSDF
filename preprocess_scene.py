import os
import numpy as np
import trimesh
import igl
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import mesh_to_sdf
from mesh_to_sdf import sample_sdf_near_surface

point_density = 500
data_dir = "data/watertight_processed_500000"
target_dir = f"data/SdfSamples/SceneNet"
deepsdf = False

filenames = [
"1Bathroom/5_labels.obj.ply",
"1Office/7_crazy3dfree_labels.obj.ply",
"1Bedroom/77_labels.obj.ply",
"1Bedroom/bedroom27.obj.ply",
"1Bedroom/bedroom_1.obj.ply",
"1Bedroom/bedroom_68.obj.ply",
"1Bedroom/bedroom_wenfagx.obj.ply",
"1Bedroom/bedroom_xpg.obj.ply",
"1Kitchen/1-14_labels.obj.ply",
"1Bathroom/29_labels.obj.ply",
"1Bathroom/107_labels.obj.ply",
"1Bathroom/28_labels.obj.ply",
"1Bathroom/4_labels.obj.ply",

"1Bathroom/69_labels.obj.ply",
"1Kitchen/2.obj.ply",
"1Living-room/living_room_33.obj.ply",
"1Office/2_hereisfree_labels.obj.ply",
"1Office/4_hereisfree_labels.obj.ply",
   

"1Bathroom/1_labels.obj.ply",
"1Bedroom/3_labels.obj.ply",
"1Kitchen/102.obj.ply",
"1Kitchen/13_labels.obj.ply",
"1Kitchen/35_labels.obj.ply",
"1Kitchen/kitchen_16_blender_name_and_mat.obj.ply",
"1Kitchen/kitchen_106_blender_name_and_mat.obj.ply",
"1Kitchen/kitchen_76_blender_name_and_mat.obj.ply",
"1Living-room/cnh_blender_name_and_mat.obj.ply",
"1Living-room/lr_kt7_blender_scene.obj.ply",
"1Living-room/pg_blender_name_and_mat.obj.ply",
"1Living-room/room_89_blender.obj.ply",
"1Living-room/room_89_blender_no_paintings.obj.ply",
"1Living-room/yoa_blender_name_mat.obj.ply",
"1Office/2_crazy3dfree_labels.obj.ply",
"1Office/4_3dmodel777.obj.ply",
]

def sample_points_on_mesh(n_points, verts, faces):
    """
    Sample points on a mesh.
    Args:
        n_points: int, number of points to sample
        verts: np.array(n_verts, 3), mesh vertices
        faces: np.array(n_faces, 3), mesh faces
    Returns:
        sampled_points: np.array(n_points, 3), sampled points
        faces_ids: np.array(n_points), ids of the faces where the points are sampled
    """
    b_coors, faces_ids = igl.random_points_on_mesh(n_points, verts, faces)
    triangles_coords = verts[faces[faces_ids]] # shape: [n points sampled, verts in triangle, 3 coords]
    sampled_points = np.sum(b_coors[:,:,None] * triangles_coords, axis=1) # some oververtices in the triangle
    return sampled_points, faces_ids


for filename in filenames:
  mesh = trimesh.load(os.path.join(data_dir, filename))
  target_fname = os.path.join(target_dir, filename+".npz")
  
#   if os.path.exists(target_fname):
#         continue
  area = mesh.area
  #n_points = int(area * point_density)
  n_points = 1000000
  print(f"sampling from {filename} with {n_points} points")
  if not deepsdf:
      # Suface & gussian noise sample:
      surface_samples, _ = sample_points_on_mesh(n_points,mesh.vertices,mesh.faces)
     
      # set standard deviation for Gaussian distribution
      sigma = 0.01

      # generate random offsets using Gaussian distribution
      noise = np.random.normal(loc=0, scale=0.1, size=surface_samples.shape)

      # add offsets to surface points
      samples = surface_samples + noise
      print("finished sampling")


      sdf,_,_ = igl.signed_distance(samples,mesh.vertices,mesh.faces,sign_type=igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER)
  
    
  if deepsdf:
      # Sample near surface
      samples, sdf = sample_sdf_near_surface(mesh, number_of_points=n_points)
      # Suface & gussian noise sample:
#       surface_samples, _ = sample_points_on_mesh(n_points,mesh.vertices,mesh.faces)
     
#       # set standard deviation for Gaussian distribution
#       sigma = 0.01

#       # generate random offsets using Gaussian distribution
#       noise = np.random.normal(loc=0, scale=0.05, size=surface_samples.shape)

#       # add offsets to surface points
#       samples = surface_samples + noise
#       print("finished sampling")
      
#       sdf = mesh_to_sdf.mesh_to_sdf(mesh, samples)
      
        
  print("finished calculating sdf")
  print(sdf.shape,np.mean(np.abs(sdf)))
  # Save as npz file
  pos = np.concatenate((samples[sdf >= 0], sdf[sdf >= 0,None]), axis=1)
  neg = np.concatenate((samples[sdf < 0], sdf[sdf < 0,None]), axis=1)
  print('pos:',pos.shape,'neg:',neg.shape)

  # Normalize the array, seems no need to normalize
 # pos_norm = (pos-np.min(pos))/(np.max(pos)-np.min(pos))
  #neg_norm = (neg-np.max(neg))/(np.max(neg)-np.min(neg))
    
  print(pos[:,-1].max(),neg[:,-1].max())
  print(pos[:,-1].min(),neg[:,-1].min())
  print(pos[:,-1].mean(),neg[:,-1].mean())
  os.makedirs(os.path.dirname(target_fname), exist_ok=True)
  np.savez(target_fname, pos=pos.astype(np.float32), neg=neg.astype(np.float32))

def as_mesh(scene_or_mesh):
    import trimesh
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

def watertight_simplify(path, name):
    import os
    import subprocess
    # Watertight
    wt_name = name + '_wt'
    bashCommand = "~/repos/Manifold/build/manifold " + os.path.join(path, name+'.obj') + " " + os.path.join(path, wt_name+'.obj')
    output = subprocess.check_output(['bash','-c', bashCommand])

    # Simplify
    wts_name = name + '_wts'
    bashCommand = "~/repos/Manifold/build/simplify -i " + os.path.join(path, wt_name+'.obj') + " -o " + os.path.join(path, wts_name+'.obj') + " -m -c 1e-2 -f 10000 -r 0.2"
    output = subprocess.check_output(['bash','-c', bashCommand])
    
    return wts_name

def write_obj(obj_config, save_name='mesh.obj'):

    """
    # Write into an obj file
    """

    v, faces, num_vertices_faces = obj_config.values()
    num_vertices = len(v)
    num_faces = len(faces)
    f = open(save_name,"w+")
    for j in range(num_vertices):
        f.write("v %.6f %.6f %.6f\n" % (v[j,0], v[j,1], v[j,2]))
    for i in range(num_faces):
        tmp_faces = faces[i,0:int(num_vertices_faces[i])]

        # one side of the face
        f.write("f ")
        for k in range(len(tmp_faces)):
            f.write("%d " % (tmp_faces[k]+1))  # index starts at 1 in obj
        f.write("\n")

        # do a reversed order to include the other side of the face
        f.write("f ")
        for k in reversed(range(len(tmp_faces))):
            f.write("%d " % (tmp_faces[k]+1))  # index starts at 1 in obj
        f.write("\n")
    f.close()
    return

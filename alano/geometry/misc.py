def convex_hull_from_points(raw_v):

    hull = ConvexHull(raw_v)
    hull_eq = hull.equations

    # H representation
    v = raw_v[hull.vertices,:]
    num_vertices = len(v)

    g = np.hstack((np.ones((num_vertices,1)), v))  # zeros for vertex
    mat = cdd.Matrix(g, number_type='fraction')
    mat.rep_type = cdd.RepType.GENERATOR
    poly = cdd.Polyhedron(mat)
    ext = poly.get_inequalities()
    out = np.array(ext)
    A = out[:,1:] # H representation
    b = out[:,0]

    # Get faces (vertices index) for obj
    num_faces = len(b)
    faces = np.ones((num_faces, 6))*-1
    num_vertices_faces = np.zeros(num_faces)
    for i in range(num_faces):
        idx = 0
        for j in range(num_vertices):
            if abs(np.dot(A[i,:], v[j,:])+b[i]) < 1e-3:
                faces[i,idx] = j
                idx = idx+1
                num_vertices_faces[i] = num_vertices_faces[i] + 1

    return {'v':v, 'faces':faces, 'num_vertices_faces':num_vertices_faces}

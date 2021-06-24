import numpy as np
from scipy import linalg



def calculate_projection_matrix(points_2d, points_3d):
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points:

                                                      [ M11      [ u1
                                                        M12        v1
                                                        M13        .
                                                        M14        .
    [ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1        M21        .
      0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1        M22        .
      .  .  .  . .  .  .  .    .     .      .       *   M23   =    .
      Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn        M24        .
      0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]      M31        .
                                                        M32        un
                                                        M33 ]      vn ]

    Then you can solve this using least squares with np.linalg.lstsq() or SVD.
    Notice you obtain 2 equations for each corresponding 2D and 3D point
    pair. To solve this, you need at least 6 point pairs.

    Args:
    -   points_2d: A numpy array of shape (N, 2)
    -   points_2d: A numpy array of shape (N, 3)

    Returns:
    -   M: A numpy array of shape (3, 4) representing the projection matrix
    """

    # Placeholder M matrix. It leads to a high residual. Your total residual
    # should be less than 1.

    ###########################################################################
    # TODO: YOUR PROJECTION MATRIX CALCULATION CODE HERE
    ###########################################################################
    # This will be my solution to this linear equation system
    # AM = B
    # A_transpose * A * M = A_transpose * B
    # M = Inverse(A_transpose * A) * (A_transpose * B)
    rhs = []
    lhs = []
    n_row, n_col = points_2d.shape
    print("Number of Pints = " + str(n_row))
    for point in range(0, n_row):
        rhs.append([points_3d[point, 0] ,points_3d[point, 1], points_3d[point, 2], 1, 0, 0, 0, 0 , -points_2d[point, 0]*points_3d[point, 0], -points_2d[point, 0]*points_3d[point, 1], -points_2d[point, 0]*points_3d[point, 2]])
        rhs.append([0, 0, 0, 0, points_3d[point, 0], points_3d[point, 1], points_3d[point, 2], 1, -points_2d[point, 1]*points_3d[point, 0], -points_2d[point, 1]*points_3d[point, 1], -points_2d[point, 1]*points_3d[point, 2]])
        lhs.append([points_2d[point, 0]])
        lhs.append([points_2d[point, 1]])
    A_T_A = np.dot(np.mat(rhs).T, np.mat(rhs))
    A_T_B = np.dot(np.mat(rhs).T, np.mat(lhs))
    RHS = np.linalg.inv(A_T_A)
    LHS = A_T_B
    M = np.reshape(np.append(np.array( np.dot(RHS, LHS).T),[1]),(3,4))
    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################
    return M

def calculate_camera_center(M):
    """
    Returns the camera center matrix for a given projection matrix.
    The center of the camera C can be found by:
        C = -Q^(-1)m4
    where your project matrix M = (Q | m4).
    Args:
    -   M: A numpy array of shape (3, 4) representing the projection matrix
    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """
    ###########################################################################
    # TODO: YOUR CAMERA CENTER CALCULATION CODE HERE
    ###########################################################################
    m = M[:, 0:3]
    m_t = M[:, 0:3].T
    M_T_mM = np.dot(-m_t, -m)
    M_T_M = np.dot(-m_t, m)
    camera_center_location = np.dot(np.linalg.inv(np.dot(-M[:,0:3].T, -M[:,0:3])), np.dot(-M[:,0:3].T, M[:,3]))
    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################
    return camera_center_location

def estimate_fundamental_matrix(points_a, points_b):
    """
    Calculates the fundamental matrix. Try to implement this function as
    efficiently as possible. It will be called repeatedly in part 3.
    You must normalize your coordinates through linear transformations as
    described on the project webpage before you compute the fundamental
    matrix.
    Args:
    -   points_a: A numpy array of shape (N, 2) representing the 2D points in
                  image A
    -   points_b: A numpy array of shape (N, 2) representing the 2D points in
                  image B
    Returns:
    -   F: A numpy array of shape (3, 3) representing the fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR FUNDAMENTAL MATRIX ESTIMATION CODE HERE
    ###########################################################################
    n_row, n_col =  points_a.shape
    n_row = points_a.shape[0]
    A = []
    B = np.ones((n_row,1))
    pa_u_mean = np.sum(points_a[:, 0])/n_row
    pa_v_mean = np.sum(points_a[:, 1])/n_row    
    pa_dis_from_mean = np.sqrt((points_a[:,0] - pa_u_mean)**2 + (points_a[:,1] - pa_v_mean)**2)
    pa_s = n_row/np.sum((pa_dis_from_mean))
    pa_s_matrix = np.array([[pa_s,0,0], [0,pa_s,0], [0,0,1]])
    pa_mean_matrix = np.array([[1,0,-pa_u_mean],[0,1,-pa_v_mean],[0,0,1]])
    a_mat = np.reshape(np.append(np.array(points_a.T),B), (3,n_row))
    a_mat = np.dot(np.dot(pa_s_matrix, pa_mean_matrix), a_mat)
    a_mat = a_mat.T

    pb_u_mean = np.sum(points_b[:,0])/n_row
    pb_v_mean = np.sum(points_b[:,1])/n_row
    pb_dis_from_mean = np.sqrt((points_b[:,0]-pb_u_mean)**2 + (points_b[:,1]-pb_v_mean)**2)
    pb_s = n_row/np.sum(pb_dis_from_mean)
    pb_s_matrix = np.array([[pb_s,0,0], [0,pb_s,0], [0,0,1]])
    pb_mean_matrix = np.array([[1,0,-pb_u_mean],[0,1,-pb_v_mean],[0,0,1]])
    T_b =np.dot(pb_s_matrix, pb_mean_matrix)
    b_mat = np.append(np.array(points_b.T),B)
    b_mat = np.reshape(b_mat, (3,n_row))
    b_mat = np.dot(T_b, b_mat)
    b_mat = b_mat.T
    
    A = []
    for i in range(0, n_row):
        A.append([a_mat[i,0]*b_mat[i,0], a_mat[i,1]*b_mat[i,0], b_mat[i,0], a_mat[i,0]*b_mat[i,1], a_mat[i,1]*b_mat[i,1], b_mat[i,1], a_mat[i,0], a_mat[i,1]])
    fund_mat = np.dot(np.linalg.inv(np.dot( np.array(A).T,  np.array(A))), np.dot( np.array(A).T, -B))
    fund_mat = np.append(fund_mat,[1])
    fund_mat = np.reshape(fund_mat,(3,3)).T
    fund_mat = np.dot(np.dot(pa_s_matrix, pa_mean_matrix).T, fund_mat)
    fund_mat = np.dot(fund_mat, T_b)
    fund_mat = fund_mat.T
    U,S,V = np.linalg.svd(fund_mat)
    S_mat = np.array([[S[0],0,0],[0,S[1],0],[0,0,0]])
    fund_mat = np.dot(U, S_mat)
    fund_mat = np.dot(fund_mat, V)
    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################
    return fund_mat

def ransac_fundamental_matrix(matches_a, matches_b):
    """
    Find the best fundamental matrix using RANSAC on potentially matching
    points. Your RANSAC loop should contain a call to
    estimate_fundamental_matrix() which you wrote in part 2.
    If you are trying to produce an uncluttered visualization of epipolar
    lines, you may want to return no more than 100 points for either left or
    right images.
    Args:
    -   matches_a: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from image A
    -   matches_b: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from image B
    Each row is a correspondence (e.g. row 42 of matches_a is a point that
    corresponds to row 42 of matches_b)
    Returns:
    -   best_F: A numpy array of shape (3, 3) representing the best fundamental
                matrix estimation
    -   inliers_a: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from image A that are inliers with
                   respect to best_F
    -   inliers_b: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from image B that are inliers with
                   respect to best_F
    """
    ###########################################################################
    # TODO: YOUR RANSAC CODE HERE
    ###########################################################################
    criteria = 0
    n_row, n_col = matches_a.shape
    number_of_iterations = 1000
    acceptable_error = 0.05
    itr_range = range(0, number_of_iterations)
    point_number_range = range(0, n_row)
    error = 0
    img_a_inlr = []
    img_b_inlr = []
    counter = 0
    for iter in itr_range:
        fund_mat = estimate_fundamental_matrix(matches_a[np.random.randint(0, n_row, size = 8), :], matches_b[np.random.randint(0, n_row, size = 8), :])
        img_a_inlr = []
        img_b_inlr = []
        counter = 0
        for point in point_number_range:
            point_a_match_list = np.append(matches_a[point, :],1)
            point_b_match_list = np.append(matches_b[point, :],1)
            error = np.dot(np.dot(point_a_match_list, fund_mat.T), point_b_match_list.T)
            abs_error = abs(error)
            if abs_error < acceptable_error:
                counter = counter + 1
                img_a_inlr.append(matches_a[point,:])
                img_b_inlr.append(matches_b[point,:])
        if counter > criteria:
            img_a_total_inlr = img_a_inlr
            img_b_total_inlr = img_b_inlr
            criteria = counter
            final_fund_mat = fund_mat
    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################
    return final_fund_mat, np.array(img_a_total_inlr), np.array(img_b_total_inlr)
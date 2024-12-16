"""Provides recovery and simulation algorithms for reflected Brownian motion with semipermeable barriers in a planar domain. 

Main methods 
-------------
* recover\_barriers\_wasserstein: 
    Recovers the barriers based on a trajectory by looking for discontinuities in the empirical transition kernel along a grid of squares. Continuity is here measured using the Wasserstein distance. 
* get\_sample\_path: 
    Simulation scheme which generates a trajectory of a reflected Brownian motion with semipermeable barriers.

Reference
------------
This module was made available and used in the paper "Recovering semipermeable barriers from reflected Brownian motion" by Alexander Van Werde and Jaron Sanders (2024). The method recover\_barriers\_wasserstein implements Algorithm 1 in that paper. 

Dependencies
------------
We rely on the Python Optimal Transport library (POT) for computing Wasserstein distances (https://pythonot.github.io/). The following libraries are also used: numpy, scipy, xml, svg, and typing.  
 """

import numpy as np 
from scipy.spatial.distance import cdist
from xml.dom import minidom
from svg.path import parse_path
from typing import List, Tuple,Literal,Callable,Dict 
from ot import emd2


### Recovery algorithm ###
def _get_truncated_wasserstein(samples_1:list, samples_2:list, truncation_level:float) -> float:
    """Returns the truncated Wasserstein distance between the empirical probability distribution associated with the given samples."""
    if (len(samples_1) == 0) or (len(samples_2) == 0):
        if len(samples_1) == len(samples_2):
            return 0
        else:
            return truncation_level
    else:            
        cost_matrix = np.minimum(cdist(samples_1,samples_2),truncation_level)
        return emd2([1/len(samples_1) for i in range(len(samples_1))],[1/len(samples_2) for i in range(len(samples_2))], cost_matrix )


def _get_empirical_transition_distributions(paths_list:list, discretization_scale:float)->Tuple[dict,Callable[[int, int], Tuple[float,float]]]:
    """The first coordinate of the output is a dictionary whose keys are (i,j) tuples parametrizing a collection of square, and whose values is a list containing the points into which the paths transitions departing from this square. The second coordinate is a function to convert the (i,j) tuples to the (x,y) coordinate of the middle of the square."""
    for i in range(len(paths_list)):
        if len(paths_list[i])<2:
            raise Exception("One of the given paths has length less than 2.")
        paths_list[i] = np.array(paths_list[i])
    xmin, ymin= np.min([np.min(path[:,0]) for path in paths_list]), np.min([np.min(path[:,1]) for path in paths_list])
    def xy_to_ij(x:float,y:float):
        return int((x - xmin)//discretization_scale), int((y-ymin)//discretization_scale) 
    def ij_to_xy(i:int,j:int):
        return xmin + discretization_scale/2 + discretization_scale*i, ymin + discretization_scale/2 + discretization_scale*j
    ij_vals = set()
    for path in paths_list:
        ij_vals = ij_vals.union({xy_to_ij(path[timestep,0],path[timestep,1]) for timestep in range(len(path)-1)})
    ij_to_dist = {ij:[] for ij in ij_vals}
    for path in paths_list:
        for timestep in range(len(path)-1):
            ij = xy_to_ij(path[timestep,0],path[timestep,1])
            ij_to_dist[ij].append((path[timestep+1,0],path[timestep+1,1]))
    return ij_to_dist,ij_to_xy
    
def get_wasserstein_change_field(paths_list:list, discretization_scale:float, truncation_level = float('inf'),comparison_range =2)->Dict[tuple, float]:
    """Determines how quickly the transition kernel varies as a function of space by discretizing the domain into squares and comparing estimated transition kernels for nearby squares. The metric used to compare the transition kernels is the truncated Wasserstein distance; see the notes below.   
    
    :param paths_list: A list of observed paths, possibly only one, from the process. Each of these paths should be a sequence of 2-tuples corresponding to the sequence of observed (x,y) coordinates. 
    :param discretization_scale: Determines the length of the sides of the squares which are used in the discretization of the domain.  
    :param truncation_level: Determines where the Euclidean metric should be thresholded in the truncated Wasserstein distance. This makes the estimate less sensitive to outliers and hence reduces false positives. 
    :param comparison_range: Determines how distant squares can be in the comparison of transition kernels. More precisely, (comparison range)^2-many comparisons are done.  
    :returns: A dictionary whose keys are (x,y) coordinates of the centers of squares, and whose values is the maximal truncated Wasserstein distance between the estimated transition kernel of that square and the kernel of a nearby square.
    
    Notes
    ------
    The truncated Wasserstein distance between two probability distributions P,Q is mathematically defined as truncated_wasserstein = inf_{X,Y} E[min(∥X-Y∥, truncation_level)] where E is the expected value and the infimum run over all couplings X,Y. (This means that X and Y have marginal distributions P and Q, but could have a nontrivial joint distribution.) 

    The default value comparison_range = 2 is chosen because the theoretical guarantees use this value. Reducing to comparison_range = 1 may also work fine in practice.     
  
    """
    distributions_dict, ij_to_xy = _get_empirical_transition_distributions(paths_list,discretization_scale)

    # To avoid comparisons with undefined values, we pad the dictionary keys a bit. 
    keys = set(distributions_dict.keys())
    new_keys = set()
    for hi in range(-comparison_range,comparison_range+1):
        for hj in range(-comparison_range,comparison_range+1):
            hkeys = {(ij[0]+hi, ij[1] +hj) for ij in keys}
            new_keys = new_keys.union(hkeys.difference(keys))
    padded_distributions_dict = {ij:[] for ij in new_keys}
    padded_distributions_dict.update(distributions_dict)

    # If required, one could make the following a more efficient by exploiting symmetry to cut down the number of hi/hj values which have to be considered.      
    change_rate_dict = {ij:0 for ij in distributions_dict.keys()}
    for hi in range(-comparison_range,comparison_range+1):
        for hj in range(-comparison_range,comparison_range+1):
            if (hi!=0 and hj!=0): 
                directional_change_rates = {ij:_get_truncated_wasserstein(distributions_dict[ij], padded_distributions_dict[(ij[0]+hi, ij[1] + hj)],truncation_level) for ij in keys}
                change_rate_dict = {ij:np.max([change_rate_dict[ij],directional_change_rates[ij]]) for ij in keys}
    return {ij_to_xy(ij[0],ij[1]):change_rate_dict[ij] for ij in keys}
 

def recover_barriers_wasserstein(paths_list:list,sensitivity_threshold:float, discretization_scale:float, truncation_level = float('inf'), comparison_range =2) -> set:
    """Recovers barriers to the stochastic process by looking for discontinuities in the transition kernel over a grid of squares. Here, continuity is measured with respect to the truncated Wasserstein distance; see the notes for the definition. 

    :param paths_list: A list of observed paths, possibly only one, from the process. Each of these paths should be a sequence of 2-tuples corresponding to the sequence of observed (x,y) coordinates. 
    :param sensitivity_threshold: Determines how much the truncated Wasserstein distance may locally vary before it is considered discontinuous behavior. 
    :param discretization_scale: Determines the length of the sides of the squares which are used in the discretization of the domain.  
    :param truncation_level: Determines where the Euclidean metric should be thresholded in the truncated Wasserstein distance. This makes the estimate less sensitive to outliers and hence reduces false positives. 
    :param comparison_range: Determines how distant squares can be in the comparison of transition kernels. More precisely, (comparison range)^2-many comparisons are done.
    :returns: The set of points where a barrier are detected. That is, where the maximal Wasserstein distance of the estimated transition kernel kernel with the kernel of a nearby square is greater than the sensitivity threshold. 

    Notes
    ------
    The truncated Wasserstein distance between two probability distributions P,Q is mathematically defined as truncated_wasserstein = inf_{X,Y} E[min(∥X-Y∥, truncation_level)] where E is the expected value and the infimum run over all couplings X,Y. (This means that X and Y have marginal distributions P and Q, but could have a nontrivial joint distribution.) 

    The default value comparison_range = 2 is chosen because the theoretical guarantees use this value. Reducing to comparison_range = 1 may also work fine in practice.  

    Example
    ----------
    .. code-block:: python
        import numpy as np
        import brownian_barrier as bb  
        import matplotlib.pyplot as plt

        # A pre-generated example sample path may be found on the github.  
        # https://github.com/Alexander-Van-Werde/Brownian-barriers/

        path = np.load("example_path.npy")
        discretization_scale = 0.015
        sensitivity_threshold = 0.0825
        detected_points = list(bb.recover_barriers_wasserstein([path],sensitivity_threshold,discretization_scale))

        # Visualize output 
        fig, ax = plt.subplots(dpi = 100)
        ax.set_aspect("equal")

        # Output 
        plt.scatter([xy[0] for xy in detected_points], [xy[1] for xy in detected_points], color = "crimson", s=0.5,label = "Detected")

        # Truth
        num_barrier_points = 500 
        angles = [2*np.pi*i/num_barrier_points for i in range(num_barrier_points)]
        barriers = np.array([[(np.cos(t),np.sin(t)) for t in angles],[(0.7*np.cos(t),0.7*np.sin(t)) for t in angles]])  
        plt.plot(barriers[0,:,0],barriers[0,:,1], color = "Black", label = "True")
        plt.plot(barriers[1,:,0],barriers[1,:,1], color = "Black", linestyle = (0, (5, 1)))
        plt.title("Detected and true barriers")
        plt.legend(loc = "lower right")
        plt.show()        
    """
    wasserstein_change_rates = get_wasserstein_change_field(paths_list,discretization_scale,truncation_level,comparison_range)
    return {xy for xy in wasserstein_change_rates.keys() if wasserstein_change_rates[xy]>=sensitivity_threshold}




### Simulation scheme ###
def _get_Brownian_piece(time_increment:float, approximation_radius:float) -> np.ndarray:
    """Returns a piece of a 2d Brownian sample path, stopped before the distance from 0 exceeds the approximation radius."""
    num_samples = int(5*approximation_radius**2/time_increment)
    increments = np.sqrt(time_increment)*np.random.normal(size = [num_samples,2])
    path = np.cumsum(increments,0)

    distances = np.array([np.linalg.norm(y) for y in path])
    if np.max(distances) <= approximation_radius:
        return path
    else: 
        index_exceeds = np.min([i for i in range(len(distances)) if distances[i]>approximation_radius])
        return path[0:index_exceeds]
    
def _get_upper_half_piece(height_start: float, time_increment:float, approximation_radius:float, local_max:float)->Tuple[np.ndarray, bool]: 
    """The first coordinate of the output is a 2d reflected Brownian motion in the upper half plane started at height height_start, stopped before the distance from 0 exceeds approximation_radius and before the local time exceeds local_max. The second coordinate indicates if the local time exceeded local_max."""
    num_samples = int(5*approximation_radius**2/time_increment)
    increments = np.sqrt(time_increment)*np.random.normal(size = [num_samples,2])
    W = np.cumsum(increments,0)
    L = -np.minimum.accumulate(np.minimum(height_start + W[:,1],0))
    Y = np.column_stack((W[:,0] , height_start + W[:,1] + L)) 

    distances = np.array([np.linalg.norm(y - np.array([0,height_start])) for y in Y])
    if np.max(distances) > approximation_radius:
        index_exceeds = np.min([i for i in range(len(distances)) if distances[i]>approximation_radius]) 
        Y = Y[0:index_exceeds]
        L = L[0:index_exceeds]
    LocalExceeds = False  
    if len(L)>0:
        if np.max(L) > local_max:
            index_exceeds = np.min([i for i in range(len(L)) if L[i]>local_max])
            Y = Y[0:index_exceeds]
            LocalExceeds = True 
    return Y, LocalExceeds 


def _get_normal(barrier:list, x0:np.ndarray)-> np.ndarray:
    """Returns an approximate unit normal to the barrier near x0. The orientation is arbitrary."""
    # Consider K = 4 nearest points to improve robustness 
    K = 4
    distances = [np.linalg.norm(y - x0) for y in barrier]
    indices = sorted(range(len(distances)), key=lambda sub: distances[sub])[:K]
    sub_barrier = [barrier[i] for i in indices]

    # Take two most distant points and use to get approximate tangent 
    distances = cdist(sub_barrier,sub_barrier)
    i,j = np.unravel_index(distances.argmax(), distances.shape)
    tangent = sub_barrier[i] - sub_barrier[j]
    tangent = tangent/np.linalg.norm(tangent)
    return np.array((-tangent[1], tangent[0]))

# The implementation of the following could be improved. 
# First, one could make it more efficient by e.g., propagating along a minimal spanning tree instead of an ad-hoc basis. 
# Second, the code is currently not robust against violations of the smoothness assumptions. Corners in the barriers sharper than 90 degrees could confuse the computations.   
def _get_normal_field(barrier:np.ndarray, approximation_radius:float)->np.ndarray:
    """Returns the vector field of inward-pointing unit normals to the barrier."""
    normal_field = np.zeros(np.shape(barrier))

    # We start by determining the orientation at the point with maximal y-value by using that the associated normal vector should point downwards. 
    i_max = np.min(np.argmax(barrier[:,1]))
    unoriented_normal =_get_normal(barrier, barrier[i_max]) 
    normal_field[i_max] = -np.sign(unoriented_normal[1])*unoriented_normal
     
    # Propagate orientation using continuity
    assigned_indices = set()
    boundary = {i_max}
    while(len(assigned_indices) + len(boundary) != len(barrier)):
        if len(boundary) == 0: 
            raise Exception("Please ensure that the spacing between the points in the barriers is greater than approximation_radius.")
        i_new = boundary.pop()
        assigned_indices.add(i_new)

        close_indices = [i for i in range(len(barrier)) if np.linalg.norm(barrier[i] - barrier[i_new]) <= approximation_radius]
        close_indices = [i for i in close_indices if (i not in assigned_indices) and (i not in boundary)]
        for i in close_indices:
            boundary.add(i)
            new_unoriented_normal =_get_normal(barrier, barrier[i])
            oriented_normal = normal_field[i_new]  
            normal_field[i] = np.sign(oriented_normal.dot(new_unoriented_normal))*new_unoriented_normal 
    return normal_field
    
def _get_orientation(x0:np.ndarray, barrier:np.ndarray, normal_field:np.ndarray)->Literal[-1, 1]:
    """Returns +1 (resp. -1) if the given point is on the positive (resp. negative) side of the given barrier."""
    distances = [np.linalg.norm(y - x0) for y in barrier]
    closest_index = np.min( np.argmin(distances))
    if (x0 - barrier[closest_index]).dot(normal_field[closest_index]) <0:
        return -1
    else: 
        return +1 


def _get_offset(barrier:list, normal:np.ndarray, x0:np.ndarray, approximation_radius:float) -> float:
    """Returns a value c such that every point y in the barrier at distance <= approximation radius from x0 satisfies y.dot(normal) < c."""
    sub_barrier = [y for y in barrier if np.linalg.norm(y - x0) <= approximation_radius] 
    if len(sub_barrier) == 0:
        raise Exception("x0 is not within distance approximation_radius from the barrier!") 
    dots = [y.dot(normal) for y in sub_barrier]
    deviation = np.max(dots) - np.min(dots)
    return np.max(dots) + deviation/10 

 
def get_sample_path(initial_condition:List[float], time_increment:float, approximation_radius: float, num_samples :int, barriers: List[list], permeabilities = list[list]) -> list:
    """Samples from a reflected Brownian motion semipermeable barriers. 

    :param initial_condition: A two-tuple indicating the point at which the process should start.
    :param time_increment: Spacing in time between the samples to be observed. This should be taken sufficiently small to get good results. (Say, smaller than approximation_radius**2/10 and smaller than 1/P with P the greatest permeability.)   
    :param approximation_radius: Parameter for the approximation scheme indicating the scale at which barriers can be approximated by straight lines. This should be taken sufficiently small to get good results. (Say, a small multiple of min\{1/kappa, rho\} where kappa is the maximal curvature of the barriers and rho is the spacing between barriers.)      
    :param num_samples: The number of samples which is desired. 
    :param barriers: A list of the lists giving the coordinates of points on the barriers. For example, barrier[0] should be a list of two-tuples of floats corresponding to the coordinates of points. 
    :param permeabilities: A list of two-tuples corresponding to the permeabilities of the two sides of each barrier. The first entry of permeabilities[0] being large makes it is easy to go from the outside to the inside of barrier[0]. Permeability zero corresponds to a non-permeable barrier.  
    :returns: A list of (x,y) coordinates of length num_samples corresponding to a sample path of reflected Brownian motion with semipermeable barriers started from initial_condition.   
    
    Notes
    ----------
    Entering barriers manually through an explicit parametric equation as in the example below can be inconvenient if non-trivial shapes are desired. A more convenient method is provided by the method svg_to_barriers.  

    Example
    ----------
    .. code-block:: python
        import numpy as np
        import brownian_barrier as bb  
        import matplotlib.pyplot as plt 

        num_barrier_points = 500 
        angles = [2*np.pi*i/num_barrier_points for i in range(num_barrier_points)]
        barriers = [[(np.cos(t),np.sin(t)) for t in angles], [(0.7*np.cos(t),0.7*np.sin(t)) for t in angles]]  

        initial_condition = (0,-0)
        permeabilities = [[0,0],[2,2]]
        observation_time = 1
        approximation_radius = 0.04
        time_increment = approximation_radius**2/50

        num_samples = int(observation_time/time_increment)
        np.random.seed(1)
        path = bb.get_sample_path(initial_condition, time_increment, approximation_radius, num_samples, barriers, permeabilities)
        path = np.array(path)

        plt.figure(figsize=(5,5))
        plt.plot(path[:,0], path[:,1], linewidth = 0.4, color = "Navy")
        for k in range(len(barriers)):
            B = np.array(barriers[k])
            if k == 0:
                plt.plot(B[:,0],B[:,1], color = "Black", linewidth = 2)
            else: 
                plt.plot(B[:,0],B[:,1], color = "Black", linewidth = 2,  linestyle = (0, (5, 1)))
        plt.show()

 
    """
    initial_condition = np.array(initial_condition)
    barriers = [np.array(B) for B in barriers]
    normal_fields = [_get_normal_field(B,approximation_radius) for B in barriers] 
    permeabilities = np.array(permeabilities)
    orientations = [_get_orientation(initial_condition,barriers[i],normal_fields[i]) for i in range(len(barriers))]
    
    if  10*time_increment > approximation_radius**2:
        raise Warning("It is advised to take time_increment is smaller than approximation_radius**2/10.")
    permeabilities_max = np.max(permeabilities)
    if permeabilities_max > 0:
        if time_increment> 1/permeabilities_max**2:
            raise Warning("It is advised to take time_increment is smaller than 1/max(permeabilities)**2.")  
    if len(permeabilities) != len(barriers): 
        raise Exception("The lenghts of the permeabilities and barriers do not match.")
    if len(barriers) <= 0:
        raise Exception("The list of barriers is empty.")
    for i in range(len(barriers)):
        if len(barriers[i]) < 20:
            raise Warning("One of the barriers has fewer than 20 points.")
        if np.min([np.linalg.norm(initial_condition - y) for y in barriers[i]]) == 0:
            raise Warning("The initial condition lies on one of the barriers. It will be taken to lie on the positive side.")  
        for j in range(len(barriers)): 
            barrier_dist = np.min(cdist(barriers[i],barriers[j]))
            if (i!=j) and (barrier_dist <= approximation_radius/2):
                raise Exception("The given value of approximation_radius is too large. Please ensure that the distance between distinct barriers is greater than 2*approximation_radius.")
    if np.max(orientations) < 0:
        raise Warning("The initial condition appears to lie on the outside of all barriers.")

    path = [initial_condition]
    current_position = initial_condition 
    while len(path) < num_samples:
        distances = [np.min([np.linalg.norm(y - current_position) for y in barrier]) for barrier in barriers]
        if np.min(distances) > approximation_radius/2:
            new_piece =current_position + _get_Brownian_piece(time_increment, np.min(distances) - approximation_radius/10)
            path = path + list(new_piece)
            if len(new_piece)>0:
                current_position = new_piece[len(new_piece)-1]
        else: 
            i = np.min([i for i in range(len(distances)) if distances[i] <= approximation_radius/2])
            B_i= barriers[i]
            if orientations[i] == +1:
                lambda_i = permeabilities[i,1]
            else: 
                lambda_i = permeabilities[i,0]
            if lambda_i >0:
                local_max = np.random.exponential(1/lambda_i)
            else:
                local_max = np.infty

            # Approximate barrier by line and sample process reflecting on this straight line
            distances = [np.linalg.norm(y - current_position) for y in B_i]
            closest_index = np.min( np.argmin(distances))

            normal = orientations[i]*normal_fields[i][closest_index] 
            offset = _get_offset(B_i,normal,current_position,approximation_radius)
            h = normal.dot(current_position) - offset  
            
            rbm_piece, LocalExceeded = _get_upper_half_piece(h,time_increment,approximation_radius,local_max) 
            rotation_matrix = np.array([[-normal[1], normal[0]], [normal[0], normal[1]]] ) 
            new_piece = current_position +  np.matmul(rotation_matrix,(rbm_piece - h*np.array([0,1])).transpose()).transpose() 

            # We check that the endpoint is on the correct side of the barrier and move it slightly otherwise.  
            if len(new_piece)>0:
                new_position = new_piece[len(new_piece)-1]
                new_orientation = _get_orientation(new_position,B_i,normal_fields[i]) 
                eps = 1/5
                epsilon = eps*np.sqrt(time_increment)
                 
                while (LocalExceeded and (new_orientation ==orientations[i])):
                    new_position = new_position - epsilon*normal  
                    new_orientation = _get_orientation(new_position,B_i,normal_fields[i]) 
                    if np.linalg.norm(new_position - current_position) > 4*approximation_radius:
                        raise Exception("Something went wrong when moving a point to the other side of a barrier!")
                while ((not LocalExceeded) and (new_orientation !=orientations[i])):
                    new_position = new_position + epsilon*normal  
                    new_orientation = _get_orientation(new_position,B_i,normal_fields[i]) 
                    if np.linalg.norm(new_position - current_position) > 4*approximation_radius:
                        raise Exception("Something went wrong! The point jumped to the other side of the barrier when it was not supposed to.")
                new_piece[len(new_piece)-1] = new_position
            
                # Update parameters. 
                path = path + list(new_piece)
                current_position = new_position
                if LocalExceeded:
                    orientations[i] = -orientations[i]
    return path[:num_samples]

def svg_to_barriers(file_path:str):
    '''Takes a .svg file and returns the coordinates of the nodes of the drawn curves.
    
    Notes
    ----------
    Suitable .svg files can be created in a vector drawing program such as inkscape. The employed procedure is as follows: (1) Open a new file and draw the desired barriers. (2) Ensure that there are enough nodes. To do this, add equally spaced points using Extensions>Modify Path>Add Nodes. (3) Save as a .svg file.     

    Example
    ----------
    .. code-block:: python
        file_path = "Barriers/egg_n_bacon.svg"
        barriers = bb.svg_to_barriers(file_path)

        plt.figure()
        for i in range(len(barriers)):
            plt.plot(barriers[i][:,0],barriers[i][:,1],   label = str(i))
        plt.legend()
        plt.show()
    '''
    doc = minidom.parse(file_path)
    barriers = []
    for ipath, path in enumerate(doc.getElementsByTagName('path')):
        B = []
        d = path.getAttribute('d')
        parsed = parse_path(d)
        for obj in parsed:
            point = (obj.start.real, -obj.start.imag)
            if point not in B:
                B.append(point)
        B = np.array(B)
        barriers.append(B) 
    doc.unlink()

    # Renormalize and recenter
    xmin = np.min([np.min(barriers[i][:,0]) for i in range(len(barriers))])
    xmax = np.max([np.max(barriers[i][:,0]) for i in range(len(barriers))])
    ymin = np.min([np.min(barriers[i][:,1]) for i in range(len(barriers))])
    ymax = np.max([np.max(barriers[i][:,1]) for i in range(len(barriers))])
    scale = np.max((xmax - xmin, ymax - ymin))

    for i in range(len(barriers)): 
        barriers[i] = np.array(barriers[i])
        barriers[i][:,0] = (barriers[i][:,0] - xmin - (xmax - xmin)/2)/scale
        barriers[i][:,1] = (barriers[i][:,1] - ymin - (ymax - ymin)/2)/scale 
    return barriers   
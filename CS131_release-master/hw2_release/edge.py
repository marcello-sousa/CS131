import numpy as np
import queue as Queue

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')
    kernel = np.flip(np.flip(kernel, axis=0), axis=1)
    ### YOUR CODE HERE
    for H in range(Hi):
        for W in range(Wi):
            out[H, W] = np.sum(kernel*padded[H:H+Hk,W:W+Wk])
    ### END YOUR CODE

    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.
    
    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp
    
    Args:
        size: int of the size of output matrix
        sigma: float of sigma to calculate kernel

    Returns:
        kernel: numpy array of shape (size, size)
    """  
    
    kernel = np.zeros((size, size))
    k = (size - 1) // 2

    ### YOUR CODE HERE
    
    factor = 1./(2*np.pi*np.power(sigma, 2))
    
    for i in range(size):
        for j in range(size):
            kernel[i, j] = factor*np.exp(-((i - k)**2 + (j - k)**2)/(2*np.power(sigma, 2)))
    
    ### END YOUR CODE

    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints: 
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: x-derivative image
    """

    out = None
    kernel = 0.5*np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]])

    ### YOUR CODE HERE
    
    out = conv(img, kernel)
    
    ### END YOUR CODE

    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints: 
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: y-derivative image
    """

    out = None
    kernel = 0.5*np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])
    ### YOUR CODE HERE
    out = conv(img, kernel)
    ### END YOUR CODE

    return out

def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W)

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W)
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W)
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)
    
    ### YOUR CODE HERE
    dx = partial_x(img)
    dy = partial_y(img)
    
    G = np.sqrt(dx**2 + dy**2)
    rad = np.arctan2(dy, dx)
    theta = 180 - rad*(180/np.pi)

    ### END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).
    
    Args:
        G: gradient magnitude image with shape of (H, W)
        theta: direction of gradients with shape of (H, W)

    Returns:
        out: non-maxima suppressed image
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45
    G = np.pad(G, 1, mode='constant')
    
    ### BEGIN YOUR CODE
    for i in range(1, H + 1):
        for j in range(1, W + 1):
            if theta[i-1, j-1] == 90 or theta[i-1, j-1] == 270:
                if G[i+1, j] <= G[i, j] and G[i-1, j] <= G[i, j]:
                    out[i-1, j-1] = G[i, j]
            elif theta[i-1, j-1] == 0 or theta[i-1, j-1] == 180:
                if G[i, j+1] <= G[i, j] and G[i, j-1] <= G[i, j]:
                    out[i-1, j-1] = G[i, j]
            elif theta[i-1, j-1] == 45 or theta[i-1, j-1] == 225:
                if G[i-1, j+1] <= G[i, j] and G[i+1, j-1] <= G[i, j]:
                    out[i-1, j-1] = G[i, j]
            elif theta[i-1, j-1] == 135 or theta[i-1, j-1] == 315:
                if G[i-1, j-1] <= G[i, j] and G[i+1, j+1] <= G[i, j]:
                    out[i-1, j-1] = G[i, j]
    ### END YOUR CODE
    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response
        high: high threshold(float) for strong edges
        low: low threshold(float) for weak edges

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values above
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values below the
            higher threshould and above the lower threshold.
    """

    strong_edges = np.zeros(img.shape)
    weak_edges = np.zeros(img.shape)

    ### YOUR CODE HERE
    strong_edges = (img >= high)
    weak_edges = (img < high) & (img >= low)        
            
    ### END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x)

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel
        H, W: size of the image
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)]
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
        if (a, b) is one of 'the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W)
        weak_edges: binary image of shape (H, W)
    Returns:
        edges: numpy array of shape(H, W)
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W))

    ### YOUR CODE HERE
    queue = Queue.Queue()
    
    for i in indices:
        queue.put(i)
        while(not queue.empty()):
            x, y = queue.get()
            edges[x, y] = 1
            index = get_neighbors(x, y, H, W)
            for j in index:
                if weak_edges[j] == 1 and edges[j] == 0:
                    queue.put(j)
    ### END YOUR CODE
    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W)
        kernel_size: int of size for kernel matrix
        sigma: float for calculating kernel
        high: high threshold for strong edges
        low: low threashold for weak edges
    Returns:
        edge: numpy array of shape(H, W)
    """
    ### YOUR CODE HERE
    kernel = gaussian_kernel(kernel_size, sigma)
    img = conv(img, kernel)
    G, theta = gradient(img)
    img = non_maximum_suppression(G, theta)
    strong_edges, weak_edges = double_thresholding(img, high, low)
    edge = link_edges(strong_edges, weak_edges)
    ### END YOUR CODE

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W)
        
    Returns:
        accumulator: numpy array of shape (m, n)
        rhos: numpy array of shape (m, )
        thetas: numpy array of shape (n, )
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    ### YOUR CODE HERE
    num_points =  len(xs)
    
    for i in range(num_points):
        for j in range(num_thetas):
            rho = xs[i] * np.cos(thetas[j]) + ys[i] * np.sin(thetas[j])
            i_x, i_y = int(rho) + diag_len, int(thetas[j]*180/np.pi) + 90
            accumulator[i_x, i_y] += 1

    ### END YOUR CODE
    
    return accumulator, rhos, thetas

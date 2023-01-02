import numpy as np

G = 1

class Particle(object):
    """
    A simple newtonian gravitating particle.
    """
    
    def __init__(self, mass=1, x=0, y=0, z=0):
        self.mass = mass
        self.position = np.array([x, y, z])
        
    def gravitate(self, other):
        disp = other.position - self.position
        norm = np.linalg.norm(disp)
        if norm == 0:
            #will return 0 force if both particles are in the same position.
            #This case is beyond the scope of this project.
            return disp
        else:
            F = G * self.mass * other.mass / (norm ** 2)
            return F * (disp / norm)

def get_forces(particles, forces=None):
    """
    Calculates forces for a list of Particles.
    Will overwrite data in forces array.
    """
    if forces is None:
        forces = np.zeros((len(particles), 3))
    for i, part in enumerate(particles):
        for j, opart in enumerate(particles[i:]):
            _f = part.gravitate(opart)
            forces[i] += _f
            forces[j] -= _f
    return forces
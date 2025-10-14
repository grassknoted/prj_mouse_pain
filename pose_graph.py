import math

class PoseGraph:
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.edges = [(0, 1), (1, 2), (2, 3), (3, 4),
                      (4, 5), (5, 0), (5, 1), (2, 4)]
        
    def compute_length(self, coord1, coord2):
        return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
    
    def compute_angle(self, coord1, coord2, coord3):
        p12 = self.compute_length(coord1, coord2)
        p23 = self.compute_length(coord2, coord3)
        p13 = self.compute_length(coord1, coord3)
        
        epsilon = 1e-10
        denominator = (2 * p12 * p23)
        if denominator < epsilon:
            return 0.0
        
        cosine_angle = (p12**2 + p23**2 - p13**2) / denominator
        cosine_angle = max(min(cosine_angle, 1), -1)
        angle = math.acos(cosine_angle)
        return math.degrees(angle)
    
    def construct_graph(self):
        lengths = {}
        angles = {}
        
        # Compute lengths for the edges
        for i, (start, end) in enumerate(self.edges, 1):
            lengths[f'l{i}'] = self.compute_length(self.coordinates[start], self.coordinates[end])
        
        # Compute angles, each theta is the angle at the first node in the tuple
        angles['theta1'] = self.compute_angle(self.coordinates[0], self.coordinates[1], self.coordinates[5])
        angles['theta2'] = self.compute_angle(self.coordinates[1], self.coordinates[0], self.coordinates[5])
        angles['theta3'] = self.compute_angle(self.coordinates[1], self.coordinates[2], self.coordinates[5])
        angles['theta4'] = self.compute_angle(self.coordinates[2], self.coordinates[1], self.coordinates[4])
        angles['theta5'] = self.compute_angle(self.coordinates[2], self.coordinates[3], self.coordinates[4])
        angles['theta6'] = self.compute_angle(self.coordinates[3], self.coordinates[2], self.coordinates[4])
        angles['theta7'] = self.compute_angle(self.coordinates[4], self.coordinates[2], self.coordinates[3])
        angles['theta8'] = self.compute_angle(self.coordinates[4], self.coordinates[2], self.coordinates[5])
        angles['theta9'] = self.compute_angle(self.coordinates[5], self.coordinates[1], self.coordinates[4])
        angles['theta10'] = self.compute_angle(self.coordinates[5], self.coordinates[0], self.coordinates[1])
        
        return [lengths[l] for l in lengths] + [angles[a] for a in angles]
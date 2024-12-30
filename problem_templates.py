import numpy as np

class ProblemTemplate:
    def __init__(self, name, description, category, default_params, generator_func):
        self.name = name
        self.description = description
        self.category = category
        self.default_params = default_params
        self.generator_func = generator_func

def generate_social_network(num_communities=3, community_size=2, inter_community_density=0.3):
    """Generate a social network partition problem."""
    total_nodes = num_communities * community_size
    adjacency_matrix = np.zeros((total_nodes, total_nodes))
    
    # Create dense connections within communities
    for c in range(num_communities):
        start_idx = c * community_size
        end_idx = start_idx + community_size
        for i in range(start_idx, end_idx):
            for j in range(i + 1, end_idx):
                adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1
    
    # Add sparse connections between communities
    for i in range(total_nodes):
        for j in range(i + 1, total_nodes):
            if i // community_size != j // community_size:  # Different communities
                if np.random.random() < inter_community_density:
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1
    
    return adjacency_matrix

def generate_course_scheduling(num_courses=4, num_time_slots=3, conflict_density=0.3):
    """Generate a course scheduling problem as graph coloring."""
    adjacency_matrix = np.zeros((num_courses, num_courses))
    
    # Generate random conflicts between courses
    for i in range(num_courses):
        for j in range(i + 1, num_courses):
            if np.random.random() < conflict_density:
                adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1
    
    return adjacency_matrix, num_time_slots

# Define problem templates
TEMPLATES = {
    "maxcut": [
        ProblemTemplate(
            name="Social Network Analysis",
            description="""
            Partition a social network into two groups to study community structure.
            - Communities: Densely connected groups
            - Inter-community links: Sparse connections between groups
            Applications: Community detection, influence analysis
            """,
            category="maxcut",
            default_params={
                "num_communities": 3,
                "community_size": 2,
                "inter_community_density": 0.3
            },
            generator_func=generate_social_network
        )
    ],
    "coloring": [
        ProblemTemplate(
            name="Course Scheduling",
            description="""
            Assign time slots to courses while avoiding conflicts.
            - Nodes: Courses
            - Edges: Conflicts (courses that can't be scheduled at the same time)
            - Colors: Available time slots
            Applications: Academic scheduling, resource allocation
            """,
            category="coloring",
            default_params={
                "num_courses": 4,
                "num_time_slots": 3,
                "conflict_density": 0.3
            },
            generator_func=generate_course_scheduling
        )
    ]
}

def get_templates(category=None):
    """Get all templates or templates for a specific category."""
    if category:
        return TEMPLATES.get(category, [])
    return {cat: templates for cat, templates in TEMPLATES.items()}

def generate_problem(template, **params):
    """Generate a problem instance from a template with given parameters."""
    # Merge default params with provided params
    actual_params = template.default_params.copy()
    actual_params.update(params)
    
    return template.generator_func(**actual_params)

try:
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    from os.path import abspath, dirname, join
    import sys

    sys.path.insert(0, "/home/ro/lab/ompl/py-bindings")
    #sys.path.insert(0, join(dirname(dirname(abspath(__file__))), 'ompl/py-bindings'))
    # sys.path.insert(0, join(dirname(abspath(__file__)), '../whole-body-motion-planning/src/ompl/py-bindings'))
    print(sys.path)
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og


class PbStateSpace(ob.RealVectorStateSpace):
    def __init__(self, num_dim) -> None:
        super().__init__(num_dim)
        self.num_dim = num_dim
        self.state_sampler = None

    def allocStateSampler(self):
        '''
        This will be called by the internal OMPL planner
        '''
        # WARN: This will cause problems if the underlying planner is multi-threaded!!!
        if self.state_sampler:
            return self.state_sampler

        # when ompl planner calls this, we will return our sampler
        return self.allocDefaultStateSampler()

    def set_state_sampler(self, state_sampler):
        '''
        Optional, Set custom state sampler.
        '''
        self.state_sampler = state_sampler


class PyOMPL():
    def __init__(self, joint_dof, joint_limits, is_state_valid) -> None:
        self.INTERPOLATE_NUM = 100
        self.DEFAULT_PLANNING_TIME = 0.001

        self.space = PbStateSpace(joint_dof)

        bounds = ob.RealVectorBounds(joint_dof)
        joint_bounds = joint_limits

        for i, (lo_bound, up_bound) in enumerate(zip(joint_bounds[0], joint_bounds[1])):
            bounds.setLow(i, lo_bound)
            bounds.setHigh(i, up_bound)
        self.space.setBounds(bounds)

        self.is_state_valid = is_state_valid

        self.ss = og.SimpleSetup(self.space)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))
        self.si = self.ss.getSpaceInformation()

        self.planner = og.BITstar(self.ss.getSpaceInformation())
        self.ss.setPlanner(self.planner)

    def plan_start_goal(self, start, goal):
        '''
        plan a path to gaol from the given robot start state
        '''
        print("start_planning")

        # set the start and goal states;
        s = ob.State(self.space)
        g = ob.State(self.space)
        for i in range(len(start)):
            s[i] = start[i]
            g[i] = goal[i]

        self.ss.setStartAndGoalStates(s, g)
        # attempt to solve the problem within allowed planning time
        solved = self.ss.solve(self.DEFAULT_PLANNING_TIME)
        res = False
        sol_path_list = []
        if solved:
            print("Found solution: interpolating into {} segments".format(self.INTERPOLATE_NUM))
            # print the path to screen
            sol_path_geometric = self.ss.getSolutionPath()
            sol_path_geometric.interpolate(self.INTERPOLATE_NUM)
            sol_path_states = sol_path_geometric.getStates()
            sol_path_list = [self.state_to_list(state) for state in sol_path_states]
            # for sol_path in sol_path_list:
            #     self.is_state_valid(sol_path)
            res = True
        else:
            print("No solution found")

        # reset robot state
        return res, sol_path_list

    def state_to_list(self, state):
        return [state[i] for i in range(6)]

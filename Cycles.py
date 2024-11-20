import numpy

class Cycles:
    def __init__(self, adj_matrix, valid_cycles):
        self.row_count = 4
        self.column_count = self.row_count
        self.action_size = self.row_count*self.row_count
        self.adj_matrix = adj_matrix
        self.valid_cycles = valid_cycles
        self.row_degree = [numpy.sum(adj_matrix[i, :]) for i in range(self.row_count)]
        self.column_degree = [numpy.sum(adj_matrix[:, i]) for i in range(self.row_count)]
        self.which_cycle = [[[] for _ in range(self.row_count)] for _ in range(self.row_count)]
        for i, valid_cycle in enumerate(self.valid_cycles):
            for k in range(len(valid_cycle)-1):
                self.which_cycle[valid_cycle[k]][valid_cycle[k+1]].append(i)
                self.which_cycle[valid_cycle[k+1]][valid_cycle[k]].append(i)
            if len(valid_cycle)>0:
                self.which_cycle[valid_cycle[len(valid_cycle)-1]][valid_cycle[0]].append(i)
                self.which_cycle[valid_cycle[0]][valid_cycle[len(valid_cycle)-1]].append(i)
        # self.which_cycle = numpy.array(self.which_cycle)
        # for i in range(len(self.which_cycle)):
        #     for k in range(len(self.which_cycle[i])):
        #         print(self.which_cycle[i][k], "   ", end=" ")
        #     print("\n")
        # print("\n\n")
    def __repr__(self):
        return "Cycles"

    def get_intial_state(self):
        return numpy.zeros((self.row_count,self.column_count))
    #action is an int relating to the cell on the game
    def get_next_state(self, state, action, player):
        row = action//self.column_count
        column = action%self.column_count
        state[row,column] = player
        #we use 2 to be an impossible move due to source/sink rules
        if numpy.sum((state[row,:]==1) | (state[row,:]==-1))==self.row_degree[row]-1:
            # print(numpy.sum((state[row,:]==1) | (state[row,:]==-1))," r=? ",self.row_degree[row]-1)
            #maybe find more efficent way to check this?
            for i in range(self.row_count):
                if state[row, i] == 0 and self.adj_matrix[row, i] == 1:
                    state[row, i] = 2
                    break
        if numpy.sum((state[:,column]==1) | (state[:,column]==-1))==self.column_degree[column]-1:
            # print(numpy.sum((state[:,column]==1) | (state[:,column]==-1)), " c=? ", self.column_degree[column]-1)
            #maybe find more efficent way to check this?
            for i in range(self.row_count):
                if state[i, column] == 0 and self.adj_matrix[i, column] == 1:
                    state[i, column] = 2
                    break
        #we use 3 to be a blocked move as marking it the other direction is now impossible
        state[column,row] = 3
        return state
    
    def get_valid_moves(self,state):
        return (self.adj_matrix.reshape(-1) - (state.reshape(-1) != 0)).astype(numpy.uint16)
    
    def check_win(self,state,action):
        
        if(action==None):
            return False
        
        row = action//self.column_count
        column = action%self.column_count
        player = state[row,column]
        
        # print("which_cycle: ", self.which_cycle[row][column])
        for i in range(len(self.which_cycle[row][column])):
            current_cycle = self.valid_cycles[i]
            # print("cur cycle: ", current_cycle)
            #checking one direction
            all_player = True
            for k in range(len(current_cycle)-1):
                if((state[current_cycle[k], current_cycle[k+1]] != player) and \
                    (state[current_cycle[k], current_cycle[k+1]] != self.get_opponent(player))):
                    all_player = False
            if(len(current_cycle)>0):
                if((state[current_cycle[len(current_cycle)-1], current_cycle[0]] != player) and \
                    (state[current_cycle[len(current_cycle)-1], current_cycle[0]] != self.get_opponent(player))):
                    all_player = False
            if(all_player):
                return True
            #checking othr direction
            all_player = True
            for k in range(len(current_cycle)-1):
                if((state[current_cycle[k+1], current_cycle[k]] != player) and \
                    (state[current_cycle[k+1], current_cycle[k]] != self.get_opponent(player))):
                    all_player = False
            if(len(current_cycle)>0):
                if((state[current_cycle[0], current_cycle[len(current_cycle)-1]] != player) and \
                    (state[current_cycle[0], current_cycle[len(current_cycle)-1]] != self.get_opponent(player))):
                    all_player = False
            if(all_player):
                return True
        
        return False
        
    
    def get_value_and_terminate(self, state, action):
        if(self.check_win(state,action)):
            return 1,True
        if(numpy.sum(self.get_valid_moves(state))==0):
            return 0,True
        return 0,False
    
    def get_opponent(self,player):
        return -player
    
    def get_opponent_value(self, value):
        return -value
    
    def change_perspective(self, state, player):
        return state * player
    
    def get_encoded_state(self, state):
        encoded_state = numpy.stack(
            (state==-1,state==0,state==1)
        ).astype(numpy.float32)
        
        #currently due to the numpy.stack call, the first dimension is len 3, 
        # where 0=-1 state, 1=0 state, 1=1 state
        # we want this to be indexed by the batch first instead of the state status,
        # so we swap the axes
        if(len(state.shape)==3):
            encoded_state = numpy.swapaxes(encoded_state, 0, 1)
        
        return encoded_state
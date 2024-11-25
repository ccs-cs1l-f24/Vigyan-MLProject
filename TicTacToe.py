import numpy

class TicTacToe:
    def __init__(self):
        self.row_count = 3
        self.column_count = 3
        self.action_size = self.row_count*self.column_count
        
    def __repr__(self):
        return "TicTacToe"

    def get_intial_state(self):
        return numpy.zeros((self.row_count,self.column_count))
    #action is an int relating to the cell on the game
    def get_next_state(self, state, action, player):
        row = action//self.column_count
        column = action%self.column_count
        state[row,column] = player
        return state
    
    def get_valid_moves(self,state):
        return (state.reshape(-1) == 0).astype(numpy.uint8)
    
    def check_win(self,state,action):
        
        if(action==None):
            return False
        
        row = action//self.column_count
        column = action%self.column_count
        player = state[row,column]

        return(
            #summing along the whole row
            numpy.sum(state[row, :]) == player*self.column_count
            or numpy.sum(state[:,column]) == player*self.row_count
            or numpy.sum(numpy.diag(state)) == player*self.row_count
            or numpy.sum(numpy.diag(numpy.flip(state,axis=0))) == player*self.row_count
        )
    
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
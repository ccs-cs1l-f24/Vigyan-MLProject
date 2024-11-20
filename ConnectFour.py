import numpy

class ConnectFour:
    def __init__(self):
        self.row_count = 6
        self.column_count = 7
        self.action_size = self.column_count
        self.in_a_row = 4
        
    def __repr__(self):
        return "ConnectFour"

    def get_intial_state(self):
        return numpy.zeros((self.row_count,self.column_count))
    #action is an int relating to the cell on the game
    def get_next_state(self, state, action, player):
        #giga brain, gives max index where 0, aka, the lowest row
        row = numpy.max(numpy.where(state[:, action]==0))
        column = action
        state[row, column] = player
        return state
    
    def get_valid_moves(self,state):
        #checking the top row, you can play if not blocked
        return (state[0]==0).astype(numpy.uint8)
    
    def check_win(self,state,action):
        
        if(action==None):
            return False
        
        row = numpy.min(numpy.where(state[:, action] != 0))
        column = action
        player = state[row,column]

        def count(rowoff, coloff):
            for i in range(1, self.in_a_row):
                r = row + (rowoff*i)
                c = action + (coloff*i)
                if(
                    r < 0 
                    or r >= self.row_count
                    or c < 0
                    or c >= self.column_count
                    or state[r][c] != player
                ):
                    #subtract by 1 to offset the sum to deal with calling it twice
                    return i -1
            return self.in_a_row -1
                
        return(
            #subtract by 1 to offset the sum to deal with calling it twice
            count(1, 0) >= self.in_a_row -1 #veritcal
            or count(1, 1) + count(-1, -1) >= self.in_a_row -1 #top left diag
            or count(1, -1) + count(-1, 1) >= self.in_a_row -1 #top right diag
            or count(0, 1) + count(0, -1) >= self.in_a_row -1 #horizontal
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
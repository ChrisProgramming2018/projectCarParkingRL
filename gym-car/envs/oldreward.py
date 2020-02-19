 if y_dif >= 2:
     227             reward = -5
     228         else:
         229             reward  = max(-dif, -5)
         230 
         231         if dif < 2:
             232             reward = 0.5**abs(dif)
             233         if dif < 0.3:
                 234             reward = 1
                 235 


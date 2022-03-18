CC = gcc
FLAG = -lm -g
OBJS = train_nn.o nn.o ndarray.o ndshape.o mnist.o loss.o layer.o grad.o activation.o
TARGET = train.out

%.o : %.c
	$(CC) -c $< $(FLAG)

main : $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(FLAG)

.PHONY : clean
clean :
	rm -f $(OBJS) $(TARGET)

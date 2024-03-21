import os
import sys
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np

def resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

class TicTacToeDeepNet(nn.Module):
    def __init__(self):
        super(TicTacToeDeepNet, self).__init__()
        self.fc1 = nn.Linear(9, 500)
        self.fc2 = nn.Linear(500, 6)
        self.fc3 = nn.Linear(6, 500)
        self.fc4 = nn.Linear(500, 9)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)
        x = F.log_softmax(x, dim = 0)
        return x

class TicTacToeBoard():
    def __init__(self):
        self.board = [0., 0., 0., 
                      0., 0., 0., 
                      0., 0., 0.]
        
    def updateBoard(self, space, pieceType):
        self.board[space] = pieceType
    
    def printBoard(self):
        board = []
        for space in self.board:
            if space == 1.:
                board.append('X')
            if space == -1.:
                board.append('O')
            if space == 0.:
                board.append('-')
        print(board[0] + "|" + board[1] + "|" + board[2])
        print(board[3] + "|" + board[4] + "|" + board[5])
        print(board[6] + "|" + board[7] + "|" + board[8])

    def checkWin(self):
        win = False
        playerwon = 0
        if (self.board[1] == self.board[0] and self.board[2] == self.board[0] and self.board[0] != 0):
            win = True
            playerwon = self.board[0]
        elif (self.board[4] == self.board[3] and self.board[5] == self.board[3] and self.board[3] != 0):
            win = True
            playerwon = self.board[3]
        elif (self.board[7] == self.board[6] and self.board[8] == self.board[6] and self.board[6] != 0):
            win = True
            playerwon = self.board[6]
        elif (self.board[3] == self.board[0] and self.board[6] == self.board[0] and self.board[0] != 0):
            win = True
            playerwon = self.board[0]
        elif (self.board[4] == self.board[1] and self.board[7] == self.board[1] and self.board[1] != 0):
            win = True
            playerwon = self.board[1]
        elif (self.board[5] == self.board[2] and self.board[8] == self.board[2] and self.board[2] != 0):
            win = True
            playerwon = self.board[2]
        elif (self.board[4] == self.board[0] and self.board[8] == self.board[0] and self.board[0] != 0):
            win = True
            playerwon = self.board[0]
        elif (self.board[4] == self.board[2] and self.board[6] == self.board[2] and self.board[2] != 0):
            win = True
            playerwon = self.board[2]
        if (playerwon == -1.):
            playerwon = 'Model'
        else:
            playerwon = 'You' 
        return win, playerwon
    
    def checkTie(self):
        for space in self.board:
            if space == 0.:
                return False
        return True
    
    def accessSpace(self, space):
        return self.board[space]
    
    def giveBoard(self):
        return self.board

board = TicTacToeBoard()

model_path = resource_path("TicTacToeModel/model.pth")

model = torch.load(model_path)

model.eval()

with torch.no_grad():
    playing = True
    while playing:
        board.printBoard()
        print("Enter a space to go from 0 to 8 (you are X, model is O) with the top left being zero and bottom right being 8:")
        space = int(input())
        if (board.accessSpace(space) != 0):
            notValid = True
            while notValid:
                print("Invalid space:")
                space = int(input())
                if (board.accessSpace(space) == 0):
                    notValid = False 
        board.updateBoard(space, 1)
        win, piece = board.checkWin()
        tie = board.checkTie()
        if (win == True):
            board.printBoard()
            print(piece + " won")
            playing = False
            break
        if (tie == True):
            board.printBoard()
            print("Tie game")
            playing = False
            break
        data = board.giveBoard()
        data = torch.tensor(data)
        output = model(data)
        output = output.argmax(0)
        output = int(output.numpy())
        if (board.accessSpace(output) != 0):
            print("model tried to go in an invalid space, ending game")
            playing = False 
        else: 
            board.updateBoard(output, -1)
            win, piece = board.checkWin()
            tie = board.checkTie()
            if (win == True):
                board.printBoard()
                print(piece + " won")
                playing = False
            if (tie == True):
                board.printBoard()
                print("Tie game")
                playing = False
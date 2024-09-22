`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    18:16:41 01/04/2022 
// Design Name: 
// Module Name:    Decoder3to8 
// Project Name: 
// Target Devices: 
// Tool versions: 
// Description: 
//
// Dependencies: 
//
// Revision: 
// Revision 0.01 - File Created
// Additional Comments: 
//
//////////////////////////////////////////////////////////////////////////////////
module Decoder3to8(
	output reg [7:0] Out,
	input wire [2:0] In,
	input wire E
    );

always @(In, E)
		if (E)
			begin
				Out[0] = &(~In);
				Out[1] = (~In[2])&(~In[1])&(In[0]);
				Out[2] = (~In[2])&(In[1])&(~In[0]);
				Out[3] = (~In[2])&(In[1])&(In[0]);
				Out[4] = (In[2])&(~In[1])&(~In[0]);
				Out[5] = (In[2])&(~In[1])&(In[0]);
				Out[6] = (In[2])&(In[1])&(~In[0]);
				Out[7] = &In;
			end
		else
			Out = 8'b00000000;

endmodule

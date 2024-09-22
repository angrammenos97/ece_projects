`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    17:58:00 01/05/2022 
// Design Name: 
// Module Name:    Mux2to1 
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
module Mux2to1(
	output reg [31:0] Out,
	input wire [63:0] In,
	input wire        Sel
    );
	 
	 always @(Sel, In)
		case (Sel)
			1'b0: Out = In[31:0];
			1'b1: Out = In[63:32];
         default: Out = 32'h00000000;			
		endcase


endmodule

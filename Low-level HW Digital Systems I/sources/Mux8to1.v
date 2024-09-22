`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    20:01:14 01/04/2022 
// Design Name: 
// Module Name:    Mux8to1 
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
module Mux8to1(
	output reg [31:0] Out,
	input wire [255:0] In,
	input wire [2:0]Sel
    );

	always @(Sel, In)
		case (Sel)
			3'b000: Out = In[31:0];
			3'b001: Out = In[63:32];
			3'b010: Out = In[95:64];
			3'b011: Out = In[127:96];
			3'b100: Out = In[159:128];
			3'b101: Out = In[191:160];
			3'b110: Out = In[223:192];
			3'b111: Out = In[255:224];		   
         default: Out = 32'h00000000;			
		endcase

endmodule

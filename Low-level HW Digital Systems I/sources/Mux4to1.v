`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    20:15:11 01/04/2022 
// Design Name: 
// Module Name:    Mux4to1 
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
module Mux4to1(
	output reg [31:0]  Out,
	input wire [127:0] In,
	input wire [1:0]   Sel
    );
	 
	 always @(Sel, In)
		case (Sel)
			2'b00: Out = In[31:0];
			2'b01: Out = In[63:32];
			2'b10: Out = In[95:64];
			2'b11: Out = In[127:96];   
         default: Out = 32'h00000000;			
		endcase


endmodule

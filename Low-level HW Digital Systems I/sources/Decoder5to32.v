`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    17:58:42 01/04/2022 
// Design Name: 
// Module Name:    Decoder5to32 
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
module Decoder5to32(
	output wire [31:0] Out,
	input wire [4:0] In,
	input wire E
    );
	 
	 wire [3:0] decSel;
	 Decoder2to4 dec24(.Out(decSel), .In(In[4:3]), .E(E));
	 Decoder3to8 dec38 [3:0](.Out(Out), .In(In[2:0]), .E(decSel));

endmodule

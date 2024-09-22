`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    18:07:10 01/04/2022 
// Design Name: 
// Module Name:    Decoder2to4 
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
module Decoder2to4(
	output reg [3:0] Out,
	input wire [1:0] In,
	input wire E
    );

	always @(In, E)
		if (E)
			case (In)
				2'b00: Out = 4'b0001;
				2'b01: Out = 4'b0010;
				2'b10: Out = 4'b0100;
				2'b11: Out = 4'b1000;
				default: Out = 4'b0000;
			endcase	//Note:produces same rtl as decoder3to8
		else
			Out = 4'b0000;

endmodule

`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    18:07:22 01/05/2022 
// Design Name: 
// Module Name:    FlipFlop 
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
module FF(
	output reg [31:0] Out,
	input wire [31:0] In,
	input wire En,
	input wire Clk,
	input wire Reset
    );
	
	always @(posedge Reset, posedge Clk)
		begin
			if(Reset)
				Out<= 32'h00000000;
		   else if(En)
				Out<= In;
		end

endmodule

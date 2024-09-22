`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   20:54:00 01/04/2022
// Design Name:   Mux4to1
// Module Name:   /home/ise/Desktop/part1/mux41_tb.v
// Project Name:  part1
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: Mux4to1
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module mux41_tb;

	// Inputs
	reg [127:0] In;
	reg [1:0] Sel;

	// Outputs
	wire [31:0] Out;

	// Instantiate the Unit Under Test (UUT)
	Mux4to1 uut (
		.Out(Out), 
		.In(In), 
		.Sel(Sel)
	);

	initial begin
		// Initialize Inputs
		In = 96'h10000000000000000;
		Sel = 2;

		

	end
      
endmodule


`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   20:41:37 01/04/2022
// Design Name:   Multiplexer32to1
// Module Name:   /home/ise/Desktop/part1/mux_tb.v
// Project Name:  part1
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: Multiplexer32to1
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module mux_tb;

	// Inputs
	reg [1023:0] In;
	reg [4:0] Sel;

	// Outputs
	wire [31:0] Out;

	// Instantiate the Unit Under Test (UUT)
	Mux32to1 uut (
		.Out(Out), 
		.In(In), 
		.Sel(Sel)
	);

	initial begin
		// Initialize Inputs
		In[1023:992] = 1;
		In = 0;
		Sel = 5'b11111;		

	end
      
endmodule


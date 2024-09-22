`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   19:13:09 01/04/2022
// Design Name:   Decoder5to32
// Module Name:   /home/ise/Desktop/part1/decoder_tb.v
// Project Name:  part1
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: Decoder5to32
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module decoder_tb;

	// Inputs
	reg [4:0] In;
	reg E;

	// Outputs
	wire [31:0] Out;

	// Instantiate the Unit Under Test (UUT)
	Decoder5to32 uut (
		.Out(Out), 
		.In(In), 
		.E(E)
	);

	initial begin
		// Initialize Inputs
		In = 5'b00000;
		E  = 1'b0;
		
		#5;
		In = 5'b00001;
		E  = 1'b1;
        
		#5
	   In = 5'b00001;
		E  = 1'b0;

      #5
      In = 5'b01001;
      E  = 1'b1;

      #5
      In = 5'b10001;

      #5
		In = 5'b11001;

	end
      
endmodule


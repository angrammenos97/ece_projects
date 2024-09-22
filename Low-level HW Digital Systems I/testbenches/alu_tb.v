`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   21:32:35 01/03/2022
// Design Name:   ALU
// Module Name:   /home/ise/Desktop/part1/alu_tb.v
// Project Name:  part1
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: ALU
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module alu_tb;

	// Inputs
	reg [31:0] A;
	reg [31:0] B;
	reg [3:0] Op;

	// Outputs
	wire [31:0] Out;
	wire Zero;

	// Instantiate the Unit Under Test (UUT)
	ALU uut (
		.Out(Out),
		.Zero(Zero), 
		.A(A), 
		.B(B), 
		.Op(Op)
	);

	initial begin
		// Initialize Inputs
		A = 32'h00000001;
		B = 32'h00000003;
		
		Op = 4'b0000;
		#5
		Op = 4'b0001;
		#5
		Op = 4'b0010;
		#5
		Op = 4'b0011;
		#5
		Op = 4'b0100;
		#5
		Op = 4'b1000;
		#5
		Op = 4'b1001;
		#5
		Op = 4'b1010;
		#5
		Op = 4'b1100;
		#5
		Op = 4'b1101;

	end
      
endmodule


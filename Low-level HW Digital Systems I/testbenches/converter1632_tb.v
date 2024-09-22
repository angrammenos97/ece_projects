`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   22:29:03 01/05/2022
// Design Name:   Converter16to32
// Module Name:   /home/ise/Desktop/part1/converter1632_tb.v
// Project Name:  part1
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: Converter16to32
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module converter1632_tb;

	// Inputs
	reg [15:0] In;
	reg [5:0] Opcode;

	// Outputs
	wire [31:0] Out;

	// Instantiate the Unit Under Test (UUT)
	Conv16to32 uut (
		.Out(Out), 
		.In(In), 
		.Opcode(Opcode)
	);

	initial begin
		// Initialize Inputs
		In = -2;
		Opcode = 6'b111000;
		#2
		Opcode = 6'b110000;
      #2
		Opcode = 6'b110010;
		#2
		Opcode = 6'b110011;
		#2
		Opcode = 6'b111111;
		#2
		Opcode = 6'b000000;
		#2
		Opcode = 6'b000001;
		#2
		Opcode = 6'b000011;
		#2
		Opcode = 6'b000111;
		#2
		Opcode = 6'b001111;
		#2
		Opcode = 6'b011111;
		
        
		// Add stimulus here

	end
      
endmodule


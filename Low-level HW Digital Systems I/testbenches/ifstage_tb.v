`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   19:18:30 01/05/2022
// Design Name:   IFSTAGE
// Module Name:   /home/ise/Desktop/part1/ifstage_tb.v
// Project Name:  part1
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: IFSTAGE
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module ifstage_tb;

	// Inputs
	reg [31:0] PC_Immed;
	reg PC_sel;
	reg Clk;
	reg Reset;
	reg PC_LdEn;

	// Outputs
	wire [31:0] Instr;

	// Instantiate the Unit Under Test (UUT)
	IFSTAGE uut (
		.Instr(Instr), 
		.PC_Immed(PC_Immed), 
		.PC_sel(PC_sel), 
		.Clk(Clk), 
		.Reset(Reset), 
		.PC_LdEn(PC_LdEn)
	);

	initial 
		begin
			// Initialize Inputs
			PC_Immed = 0;
			PC_sel = 0;
			Clk = 0;
			Reset = 1;
			PC_LdEn = 1;

			#2
			Reset = 0;
			
			#16
			PC_Immed = -24;
			PC_sel = 1;
			
			#2
			PC_sel = 0;
			
		end
	
	always
		begin
			#1 Clk = ~Clk;
		end
      
endmodule


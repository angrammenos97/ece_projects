`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   16:26:46 01/05/2022
// Design Name:   RF
// Module Name:   /home/ise/Desktop/part1/rf_tb.v
// Project Name:  part1
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: RF
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module rf_tb;

	// Inputs
	reg [31:0] Din;
	reg [4:0] Ard1;
	reg [4:0] Ard2;
	reg [4:0] Awr;
	reg CLK;
	reg WrEn;

	// Outputs
	wire [31:0] Dout1;
	wire [31:0] Dout2;

	// Instantiate the Unit Under Test (UUT)
	RF uut (
		.Dout1(Dout1), 
		.Dout2(Dout2), 
		.Din(Din), 
		.Ard1(Ard1), 
		.Ard2(Ard2), 
		.Awr(Awr), 
		.CLK(CLK), 
		.WrEn(WrEn)
	);

	initial 
		begin
			// Initialize Inputs
			Din = 4;
			Ard1 = 0;
			Ard2 = 1;
			Awr = 0;
			CLK = 0;
			WrEn = 0;
			
			#10
			WrEn = 1;
			
			#10
			Awr = 1;
			Din = 5;
			
			#10
			WrEn = 0;
			
		end
		
	always
		begin
			#5 CLK = ~CLK;
		end

      
endmodule


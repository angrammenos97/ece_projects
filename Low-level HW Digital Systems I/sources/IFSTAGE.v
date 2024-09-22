`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    18:15:37 01/05/2022 
// Design Name: 
// Module Name:    IFSTAGE 
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
module IFSTAGE(
	input		wire [31:0] PC_Immed,
	input		wire PC_sel, PC_LdEn, Reset, Clk,
	output	wire [31:0] Instr
    );

   wire [31:0] PC_Out, mux21_Out, PC_4, PC_4_Immed;
	
	Mux2to1 mux21(.Out(mux21_Out), .In({PC_4_Immed, PC_4}), .Sel(PC_sel));
	FF PC(.Out(PC_Out), .In(mux21_Out), .Reset(Reset), .En(PC_LdEn), .Clk(Clk));
	IMEM imem_module(.addr(PC_Out[11:2]), .dout(Instr), .clk(Clk));	
	
	assign PC_4       = PC_Out + 4;
	assign PC_4_Immed = PC_4   + PC_Immed;	

endmodule

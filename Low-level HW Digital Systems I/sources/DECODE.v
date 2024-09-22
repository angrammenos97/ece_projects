`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    11:19:23 01/06/2022 
// Design Name: 
// Module Name:    DECODE 
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
module DECODE(
	input  wire [31:0] Instr,
	input  wire RF_WrEn,
	input  wire [31:0] ALU_out, MEM_out,
	input  wire RF_WrData_sel, RF_B_sel, Clk,
   output wire [31:0] Immed, RF_A, RF_B
	);
	 
	 wire [31:0] RfWrData, Ard2Rf;
	 wire [4:0]  Ard2Rf_muxed;
	 
	 Mux2to1  readRegMux(.Out(Ard2Rf), .In({{27'b0, Instr[20:16]}, {27'b0, Instr[15:11]}}), .Sel(RF_B_sel));
	 Mux2to1 RFWrDataMux(.Out(RfWrData), .In({MEM_out, ALU_out}), .Sel(RF_WrData_sel));
	 
	 Conv16to32 Immed16to32(.Out(Immed), .In(Instr[15:0]), .Opcode(Instr[31:26]));
	 
	 RF RfMem(.Dout1(RF_A), .Dout2(RF_B), .Din(RfWrData), .Ard1(Instr[25:21]), .Ard2(Ard2Rf_muxed), .Awr(Instr[20:16]), .CLK(Clk), .WrEn(RF_WrEn));

    // get only five LS bits from mux
	 assign Ard2Rf_muxed = Ard2Rf[4:0];
	 
endmodule

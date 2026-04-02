import React from 'react';
import { Tabs } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';

import { useTheme } from '@/theme';

type TabIconProps = {
  color: string;
  size: number;
};

export default function TabLayout() {
  const theme = useTheme();

  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarActiveTintColor: theme.colors.primary,
        tabBarInactiveTintColor: theme.colors.textMuted,
        tabBarStyle: {
          backgroundColor: theme.colors.card,
          borderTopColor: theme.colors.border,
          height: 72,
          paddingBottom: 8,
          paddingTop: 8,
        },
        tabBarLabelStyle: {
          fontSize: 10,
          fontWeight: '700',
          letterSpacing: 0.2,
        },
      }}
    >
      <Tabs.Screen
        name="dashboard"
        options={{
          title: 'Dashboard',
          tabBarIcon: ({ color, size }: TabIconProps) => <Ionicons name="stats-chart-outline" size={size} color={color} />,
        }}
      />
      <Tabs.Screen
        name="runs"
        options={{
          title: 'Runs',
          tabBarIcon: ({ color, size }: TabIconProps) => <Ionicons name="pulse-outline" size={size} color={color} />,
        }}
      />
      <Tabs.Screen
        name="datasets"
        options={{
          title: 'Datasets',
          tabBarIcon: ({ color, size }: TabIconProps) => <Ionicons name="layers-outline" size={size} color={color} />,
        }}
      />
      <Tabs.Screen
        name="models"
        options={{
          title: 'Models',
          tabBarIcon: ({ color, size }: TabIconProps) => <Ionicons name="flask-outline" size={size} color={color} />,
        }}
      />
      <Tabs.Screen
        name="more"
        options={{
          title: 'More',
          tabBarIcon: ({ color, size }: TabIconProps) => <Ionicons name="ellipsis-horizontal-outline" size={size} color={color} />,
        }}
      />
    </Tabs>
  );
}
